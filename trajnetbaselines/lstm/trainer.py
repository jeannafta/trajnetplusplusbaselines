"""Command line tool to train an LSTM model."""

import argparse
import datetime
import logging
import time
import random
import os 
import numpy as np
import torch
import trajnettools

from .. import augmentation
from .loss import PredictionLoss, L2Loss
from .lstm import LSTM, LSTMPredictor, drop_distant
from .pooling import Pooling, HiddenStateMLPPooling
from .. import __version__ as VERSION


class Trainer(object):
    def __init__(self, model=None, criterion=None, optimizer=None, lr_scheduler=None,
                 device=None):
        self.model = model if model is not None else LSTM()
        # self.criterion = criterion if criterion is not None else PredictionLoss()
        self.criterion = criterion if criterion is not None else L2Loss()        
        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 15))

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)

    def loop(self, train_scenes, val_scenes, out, epochs=35, start_epoch=0):
        for epoch in range(start_epoch, start_epoch + epochs):
            state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.lr_scheduler.state_dict()}
            LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch))
            self.train(train_scenes, epoch)
            self.val(val_scenes, epoch)


        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        LSTMPredictor(self.model).save(state, out + '.epoch{}'.format(epoch))
        LSTMPredictor(self.model).save(state, out)

    def lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def is_nonlinear_for_final_point(self, scene, distance_threshold=1.0):
        # check linear extrapolation is off enough to make this interesting
        primary_path = scene[:, 0]
        linear_prediction = primary_path[8] + 12.0 / 3.0 * (primary_path[8] - primary_path[5])
        prediction_diff = primary_path[20] - linear_prediction
        d2 = prediction_diff[0]**2 + prediction_diff[1]**2
        # print(d, primary_path, linear_prediction)
        return d2 > distance_threshold**2

    def is_nonlinear(self, scene):
        primary_path = scene[:, 0]
        primary_velocities = primary_path[1:] - primary_path[:-1]
        primary_accelerations = primary_velocities[1:] - primary_velocities[:-1]
        primary_accelerations2 = np.sum(np.square(primary_accelerations), axis=1)
        distances_2 = np.sum(np.square(scene[:, 1:] - scene[:, 0:1]), axis=2)
        for acc2, d2 in zip(primary_accelerations2, distances_2):
            if acc2 < 0.1**2:
                continue

            smallest_distance2 = np.nanmin(d2)
            if smallest_distance2 != smallest_distance2:
                continue
            if smallest_distance2 > 2.0**2:
                continue

            return True

        return False

    def train(self, scenes, epoch):
        start_time = time.time()

        print('epoch', epoch)
        self.lr_scheduler.step()

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        for scene_i, (_, scene) in enumerate(scenes):
            scene_start = time.time()
            scene = drop_distant(scene)

            # if self.is_nonlinear(scene) and random.random() < 0.9:
            #     # print('drop')
            #     continue
            # print('continue')

            scene = augmentation.random_rotation(scene)
            scene = torch.Tensor(scene).to(self.device)
            preprocess_time = time.time() - scene_start

            loss = self.train_batch(scene)
            epoch_loss += loss
            total_time = time.time() - scene_start

            if scene_i % 10 == 0:
                self.log.info({
                    'type': 'train',
                    'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                    'time': round(total_time, 3),
                    'data_time': round(preprocess_time, 3),
                    'lr': self.lr(),
                    'loss': round(loss, 3),
                })

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / len(scenes), 3),
            'time': round(time.time() - start_time, 1),
        })

    def val(self, val_scenes, epoch):
        val_loss = 0.0
        eval_start = time.time()
        self.model.train()  # so that it does not return positions but still normals
        for _, scene in val_scenes:
            scene = drop_distant(scene)
            scene = torch.Tensor(scene).to(self.device)
            val_loss += self.val_batch(scene)
        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / len(val_scenes), 3),
            'time': round(eval_time, 1),
        })

    def train_batch(self, xy):
        # augmentation: random coordinate shifts
        # noise = (torch.rand_like(xy) - 0.5) * 2.0 * 0.03
        # xy += noise

        # augmentation: random stretching
        # x_scale = 0.9 + 0.2 * random.random()
        # y_scale = 0.9 + 0.2 * random.random()
        # xy[:, :, 0] *= x_scale
        # xy[:, :, 1] *= y_scale

        observed = xy[:9]
        prediction_truth = xy[9:-1].clone()  ## CLONE
        targets = xy[2:, 0] - xy[1:-1, 0]

        # augmentation: vary the length of the observed data a bit
        # truncate_n = int(random.random() * 4.0)
        # observed = observed[truncate_n:]
        # targets = targets[truncate_n:]

        self.optimizer.zero_grad()
        rel_outputs, outputs = self.model(observed, prediction_truth)

        loss = self.criterion(rel_outputs[:, 0], targets) * 100
        loss.backward()

        self.optimizer.step()
        return loss.item()

    def val_batch(self, xy):
        observed = xy[:9]
        prediction_truth = xy[9:-1].clone()  ## CLONE

        with torch.no_grad():
            rel_outputs, outputs = self.model(observed, prediction_truth)

            targets = xy[2:, 0] - xy[1:-1, 0]
            loss = self.criterion(rel_outputs[:, 0], targets) * 100

        return loss.item()


def main(epochs=30):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp'),
                        help='type of LSTM to train')
    parser.add_argument('--train-input-files', default='data/train/**/*.ndjson',
                        help='glob expression for train input files')
    parser.add_argument('--val-input-files', default='data/val/**/*.ndjson',
                        help='glob expression for validation input files')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--path', default='trajnew',  
                        help='glob expression for data files')

    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--pre-epochs', default=5, type=int,
                          help='number of epochs')
    pretrain.add_argument('--pre-lr', default=1e-3, type=float,
                          help='initial learning rate')
    pretrain.add_argument('--pre-lr-div-schedule', default=3, type=int,
                          help='pre-training learning rate division schedule')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')
    pretrain.add_argument('--drop-input-embedding', default=False, action='store_true',
                          help='drop input embedding before loading state')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='RNN hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')

    args = parser.parse_args()

    # set model output file
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    # if args.output is None:
    #     args.output = 'output/{}_lstm_{}.pkl'.format(args.type, timestamp)
    if not os.path.exists('OUTPUT_BLOCK/{}'.format(args.path)):
        os.makedirs('OUTPUT_BLOCK/{}'.format(args.path))
    if args.output:
        args.output = 'OUTPUT_BLOCK/{}/{}_{}.pkl'.format(args.path, args.type, args.output)  
    else:
        args.output = 'OUTPUT_BLOCK/{}/{}.pkl'.format(args.path, args.type) 
        
    # configure logging
    from pythonjsonlogger import jsonlogger
    import socket
    import sys
    if args.load_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')        
    file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    # read in datasets
    args.path = 'DATA_BLOCK/' + args.path

    train_scenes = list(trajnettools.load_all(args.path + '/train/**/*.ndjson'))
    val_scenes = list(trajnettools.load_all(args.path + '/val/**/*.ndjson'))

    # create model
    pool = None
    if args.type == 'hiddenstatemlp':
        pool = HiddenStateMLPPooling(hidden_dim=args.hidden_dim)
    elif args.type != 'vanilla':
        pool = Pooling(type_=args.type, hidden_dim=args.hidden_dim)
    model = LSTM(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim)

    # train
    if args.load_state:
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] 
        trainer = Trainer(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device)
        trainer.loop(train_scenes, val_scenes, args.output, epochs=args.epochs, start_epoch=start_epoch)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        trainer = Trainer(model, optimizer=optimizer, device=args.device)
        trainer.loop(train_scenes, val_scenes, args.output, epochs=args.epochs)


if __name__ == '__main__':
    main()
