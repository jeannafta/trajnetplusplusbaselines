import math
import torch
import torch.nn as nn

class SocialNCE():
    '''
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    '''
    def __init__(self, obs_length, pred_length, head_projection, encoder_sample, temperature, horizon, sampling):  # ENCODER SAMPLE: python -m trajnetbaselines.lstm.trainer --path synth_data --type 'directional' --goals --augment --contrast_weight 1
                                                                                                                   # EVENT ENCODER: python -m trajnetbaselines.lstm.trainer --path synth_data --type 'directional' --goals --augment --contrast_weight 1 --contrast_sampling "multi"
        # problem setting
        self.obs_length = obs_length     #  9 by default  (number of observation time steps) 
        self.pred_length = pred_length   # 12 by default  (number of prediction time steps) 

        # nce models
        self.head_projection = head_projection   # psi (projection  head: instance of the class ProjHead)
        self.encoder_sample = encoder_sample     # phi (space encoder or event encoder: instances of classes EventEncoder or SpatialEncoder)

        # nce loss
        self.criterion = nn.CrossEntropyLoss()

        # nce param
        self.temperature = temperature   # 0.07 by default
        self.horizon = horizon   # 4 

        # sampling param
        self.sampling = sampling
        self.noise_local = 0.1  # noise epsilon 
        self.min_seperation = 0.2  # rho = minimum physical distance between two agents
        self.agent_zone = self.min_seperation * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])   # we obtain the comfort zone of the agents: the 8+1 points represent points of the trigonometric circle + center of the circle. 
        
        self.max_seperation = 5. # dmax
        
    def spatial(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            Input:
                batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
                batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
                batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]  --> h  
            Output:
                loss: social nce loss
        '''

        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------
        # a batch is composed of 8 scenes which can be found with the help of the batch_split tensor
        
        for i in range(batch_split.shape[0] - 1):
            traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
            traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
            plot_scene(traj_primary, traj_neighbor, fname='scene_{:d}.png'.format(i))
            # import pdb; pdb.set_trace()

        # #####################################################
        #           TODO: fill the following code
        # #####################################################
        torch.autograd.set_detect_anomaly(True)

            
        # get the ID of the primary agent of the 8 sub scenes
        primary_agents_ID = batch_split[:-1]
        
        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------

        # sampling
        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split)
        
        ## test plot samples
        #for i in range(batch_split.shape[0] - 1):
        #    traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
        #    traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
        #    plot_samples(traj_primary, traj_neighbor, fname='scene_and_samples{:d}.png'.format(i), 
        #                 sample_pos_scene=sample_pos[i], sample_neg_scene=sample_neg[i],                                    obs_length=self.obs_length, pred_length=self.pred_length)
        #import pdb; pdb.set_trace()
            
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------

        # compute query (old method of milestone 2)
        emb_obsv = self.head_projection(batch_feat[0, primary_agents_ID, :])     # [8, 128] --> psy --> [8, 8]
        query = nn.functional.normalize(emb_obsv, dim=1)                         # [8, 8]
        
        # compute keys 
        emb_pos = self.encoder_sample(sample_pos)   # [8, 2] --> phi --> [8, 8]
        emb_neg = self.encoder_sample(sample_neg)   # [8, (M-1)*9, 2] --> phi --> [8, (M-1)*9, 8]

        key_pos = nn.functional.normalize(emb_pos, dim=1)  # [8, 8]
        key_neg = nn.functional.normalize(emb_neg, dim=2)  # [8, (M-1)*9, 8]

        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------

        sim_pos = (query * key_pos).sum(dim=1)  # [8]
        sim_neg = (query[:, None, :] * key_neg).sum(dim=2)     # [8, (M-1)*9]
        
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
        
        # logits
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature  # to obtain the logit vector of slide 89 (Lecture 8)
        
        labels = torch.zeros(logits.size(0), dtype=torch.long)
        
        loss = self.criterion(logits, labels)
        
        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        '''

        torch.autograd.set_detect_anomaly(True)
            
        # get the ID of the primary agent of the 8 sub scenes
        primary_agents_ID = batch_split[:-1]
        
        # -----------------------------------------------------
        #               Contrastive Sampling
        # -----------------------------------------------------

        #event_sampling
        sample_pos, sample_neg = self._event_sampling(batch_scene, batch_split) # (8, 12, 2) and (8, 12, 9*(M-1), 2)

        # -----------------------------------------------------
        #              Lower-dimensional Embedding
        # -----------------------------------------------------
        
        # compute query (new method explained in the milestone 3 part of the ReadMe)
        emb_obsv = self.head_projection(batch_feat[:self.pred_length-self.horizon+1, primary_agents_ID, :])     # [9, 8, 128] --> psy --> [9, 8, 8]
        query = nn.functional.normalize(emb_obsv, dim=2)      # [9, 8, 8] [time, batch, hidden_dim]
        query = torch.repeat_interleave(query, repeats=self.horizon, dim=0)  # [4*9, 8, 8]
        query = query.permute(1, 0, 2) # [8, 4*9, 8]
        
        #compute time_pos and time_neg
        time_pos = (torch.ones(sample_pos.size(0))[:, None] * (torch.arange(self.pred_length) - (self.pred_length-1)*(0.5))[None, :]) # (num_scene, pred_length) [8, 12]
        time_neg = (torch.ones(sample_neg.size(0), sample_neg.size(2))[:, None, :] * (torch.arange(self.pred_length) - (self.pred_length-1)*(0.5))[None, :, None]) # (num_scene, pred_length, 9*(M-1)) [8, 12, 9*(M-1)]

        # compute keys
        emb_pos = self.encoder_sample(sample_pos, time_pos[:,:,None])   # (num_scene, horizon, head_dim) [8, 12, 8]
        emb_neg = self.encoder_sample(sample_neg, time_neg[:,:,:,None])   # (num_scene, horizon, num_neighbors*9, head_dim) [8, 12, (M-1)*9, 8]

        key_pos = nn.functional.normalize(emb_pos, dim=2)  # [8, 12, 8]
        key_neg = nn.functional.normalize(emb_neg, dim=3)  # [8, 12, (M-1)*9, 8]
        
        # we call the get_indices function to get the indices in the right order to compute similarities 
        indices = get_indices(self.horizon, self.pred_length)
        key_pos_sim = key_pos[:, indices, :]  # [8, 36, 8]
        key_neg_sim = key_neg[:, indices, :, :]  # [8, 36, (M-1)*9, 8]
        
        # -----------------------------------------------------
        #                   Compute Similarity
        # -----------------------------------------------------
        
        sim_pos = (query * key_pos_sim).sum(dim=2)  # [8, 36]
        sim_neg = (query[:, :, None, :] * key_neg_sim).sum(dim=3) # [8, 36, 9*(M-1)]
        
        # -----------------------------------------------------
        #                       NCE Loss
        # -----------------------------------------------------
        flat_pos = sim_pos.view(-1).unsqueeze(1) # [288, 1]
        flat_neg = sim_neg.view(sim_neg.size(0)*sim_neg.size(1), -1) # [288, 9*(M-1)]
        
        # logits
        logits = torch.cat([flat_pos, flat_neg], dim=1) / self.temperature  # to obtain the logit vector of slide 89 (Lecture 8)

        labels = torch.zeros(logits.size(0), dtype=torch.long)
        
        loss = self.criterion(logits, labels)

        return loss
        
    def _sampling_spatial(self, batch_scene, batch_split):

        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]

        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        batch_size = batch_split.shape[0] - 1
        
        # first we need to find the scene which has the maximum number of neigbours so that we can return at the end of the function a tensor which contains the negative samples of all the scenes although the scenes of the batch don't have the same number of neigbhours (if a scene has a number of neigbours lower than the maximum number of neigbours then we will fill the missing values with Nans)
        
        max_neigh = 0
        for i in range(batch_split.shape[0] - 1):
            max_neigh = max(max_neigh, batch_split[i+1]-(batch_split[i]+1))
        
        # initialize sample_pos, sample_neg
        sample_pos = torch.full((batch_size,2), float("nan")).float()
        sample_neg = torch.full((batch_size,max_neigh*self.agent_zone.shape[0],2), float("nan")).float()
        
        
        # now let's make a loop over the scenes to get for each scene the positive sample and the (M_i-1)*9 negative ones 
        for i in range(batch_size):
            
            # batch_split[i] is the primary agent ID
            # [batch_split[i]+1:batch_split[i+1]] list of agent neighbours ID

            primary_ID = batch_split[i]
            neighbours_ID = range(batch_split[i]+1, batch_split[i+1])

            gt_future_primary = gt_future[:, primary_ID, :]   # [pred_length, 1 primary, 2]
            gt_future_neigbours =  gt_future[:, neighbours_ID, :] # [pred_length, M-1 neigbours, 2]
            
            # if the primary agent has no neighbours in the scene, we skip the scene 
            if gt_future_neigbours.shape[1]==0:
                continue
                
        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------
         
        
            sample_pos_scene = gt_future_primary[self.horizon-1]          # [2]  vector of dimention 2*1 which gives the position of the positive sample 
            sample_pos_scene += torch.randn(size=sample_pos_scene.size()) * self.noise_local   # [2]
        
        
        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------
        
            sample_neg_scene = gt_future_neigbours[self.horizon-1]     # [M-1, 2]
            # need to reshape the tensor so that we have 9 negative samples in total around each sample seed [(M-1)*9, 2]
            sample_neg_scene = torch.repeat_interleave(sample_neg_scene, self.agent_zone.shape[0], dim=0)    # [(M-1)*9, 2]
            agent_zone = torch.cat([self.agent_zone]*gt_future_neigbours.shape[1])    # [(M-1)*9, 2]
            sample_neg_scene += agent_zone
            sample_neg_scene += torch.randn(size=sample_neg_scene.size()) * self.noise_local     # [(M-1)*9, 2]
            
            
        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

            dist = sample_pos_scene - sample_neg_scene
            sample_neg_scene[dist.norm(dim=1)<self.min_seperation] = torch.full(sample_neg_scene[dist.norm(dim=1)<self.min_seperation].size(), float("nan"))
            
        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------
        
            sample_neg_scene[dist.norm(dim=1)>self.max_seperation] = torch.full(sample_neg_scene[dist.norm(dim=1)>self.max_seperation].size(), float("nan"))
        
        # ---------------------------------------------------------------
        #  Storage of samples from each scene in the tensor for the batch  
        # ---------------------------------------------------------------
        
            sample_pos[i,:]=sample_pos_scene
            sample_neg[i,range(len(neighbours_ID)*self.agent_zone.shape[0]),:]=sample_neg_scene
            
        # -----------------------------------------------------
        #      Set nan data to -10. to allow gradient flow 
        # -----------------------------------------------------
        
        sample_pos[torch.isnan(sample_pos)] = -10.  # nans because of missing data
        sample_neg[torch.isnan(sample_neg)] = -10.  # nans because of missing data or number of neigbhours < max number of neigbhours or if dist is out of the range [dmin, dmax]
        
        return sample_pos, sample_neg 


    def _event_sampling(self, batch_scene, batch_split):
        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]

        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        batch_size = batch_split.shape[0] - 1

         # first we need to find the scene which has the maximum number of neigbours so that we can return at the end of the function a tensor which contains the negative samples of all the scenes although the scenes of the batch don't have the same number of neigbhours (if a scene has a number of neigbours lower than the maximum number of neigbours then we will fill the missing values with Nans)
        max_neigh = 0
        for i in range(batch_split.shape[0] - 1):
            max_neigh = max(max_neigh, batch_split[i+1]-(batch_split[i]+1))

        # initialize sample_pos, sample_neg
        sample_pos = torch.full((batch_size, self.pred_length, 2), float("nan")).float() # [8, 12, 2]
        sample_neg = torch.full((batch_size, self.pred_length, max_neigh*self.agent_zone.shape[0], 2), float("nan")).float() # [8, 12, 9*(M-1), 2]


        # now let's make a loop over the scenes to get for each scene the positive sample and the (M_i-1)*9 negative ones
        for i in range(batch_size):

            # batch_split[i] is the primary agent ID
            # [batch_split[i]+1:batch_split[i+1]] list of agent neighbours ID

            primary_ID = batch_split[i]
            neighbours_ID = range(batch_split[i]+1, batch_split[i+1])

            gt_future_primary = gt_future[:, primary_ID, :]   # [pred_length, 1 primary, 2]
            gt_future_neigbours =  gt_future[:, neighbours_ID, :] # [pred_length, M-1 neigbours, 2]
            
            # if the primary agent has no neighbours in the scene, we skip the scene 
            if gt_future_neigbours.shape[1]==0:
                continue

        # -----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------


            sample_pos_scene = gt_future_primary[0:self.pred_length]          # [12, 2]  vector of dimention [4,2] which gives the position of the positive sample
            sample_pos_scene += torch.randn(size=sample_pos_scene.size()) * self.noise_local   # [4, 2]

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------

            sample_neg_scene = gt_future_neigbours[0:self.pred_length]     # [12, M-1, 2]
            # need to reshape the tensor so that we have 9 negative samples in total around each sample seed [4 ,(M-1)*9, 2]
            sample_neg_scene = torch.repeat_interleave(sample_neg_scene, self.agent_zone.shape[0], dim=1)    # [4, (M-1)*9, 2]
            agent_zone = torch.cat([self.agent_zone]*gt_future_neigbours.shape[1])    # [(M-1)*9, 2]
            agent_zone = agent_zone[None, ...].repeat(self.pred_length, 1, 1) # [ 9*(M-1), 2, 4]
            sample_neg_scene += agent_zone
            sample_neg_scene += torch.randn(size=sample_neg_scene.size()) * self.noise_local     # [4, (M-1)*9, 2]


        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

            dist = (sample_pos_scene[:, None, :] - sample_neg_scene).norm(dim=2)
            sample_neg_scene[dist<self.min_seperation] = torch.full(sample_neg_scene[dist<self.min_seperation].size(), float("nan"))

        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------

            sample_neg_scene[dist>self.max_seperation] = torch.full(sample_neg_scene[dist>self.max_seperation].size(), float("nan"))

        # ---------------------------------------------------------------
        #  Storage of samples from each scene in the tensor for the batch
        # ---------------------------------------------------------------

            sample_pos[i,:,:]=sample_pos_scene
            sample_neg[i,:,range(len(neighbours_ID)*self.agent_zone.shape[0]),:]=sample_neg_scene

        # -----------------------------------------------------
        #      Set nan data to -10. to allow gradient flow
        # -----------------------------------------------------

        sample_pos[torch.isnan(sample_pos)] = -10.  # nans because of missing data
        sample_neg[torch.isnan(sample_neg)] = -10.  # nans because of missing data or number of neigbhours < max number of neigbhours or if dist is out of the range [dmin, dmax]

        return sample_pos, sample_neg 

class EventEncoder(nn.Module):
    '''
        Event encoder that maps an sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):

        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out

class SpatialEncoder(nn.Module):
    '''
        Spatial encoder that maps an sampled location to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state):
        return self.encoder(state)

class ProjHead(nn.Module):
    '''
        Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),   
            nn.Linear(hidden_dim, head_dim)
            )

    def forward(self, feat):
        return self.head(feat)

def plot_scene(primary, neighbor, fname):
    '''
        Plot raw trajectories
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:, 0], primary[:, 1], 'k-')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')

    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
def plot_samples(primary, neighbor, fname, sample_pos_scene, sample_neg_scene, obs_length, pred_length):
    '''
    Plot raw trajectories and positive/negative samples
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:obs_length+1, 0], primary[:obs_length+1, 1], 'k')
    ax.plot(primary[obs_length:obs_length+pred_length, 0], primary[obs_length:obs_length+pred_length, 1], 'k--')
    ax.scatter(sample_pos_scene[0], sample_pos_scene[1], marker='*', color='green')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:obs_length+1, i, 0], neighbor[:obs_length+1, i, 1], 'b')
        ax.plot(neighbor[obs_length:obs_length+pred_length, i, 0], neighbor[obs_length:obs_length+pred_length, i, 1], 'b--')
    
    for i in range(sample_neg_scene.shape[0]):
        ax.scatter(sample_neg_scene[i, 0], sample_neg_scene[i, 1], marker='*', color='red')

    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def get_indices(horizon, pred_length):
    '''
    Get the indices for improved technique of milestone 3
    '''
    liste = list(range(horizon))*(pred_length-horizon+1)
    for index in range(len(liste)):
        liste[index]+=math.floor(index/horizon)
    return liste 