
TrajNet++ : The Trajectory Forecasting Framework
================================================

Milestone 1

Group A:
Gaelle Abi Younes
Jean Naftalski
Florent Zolliker

1. Comparaison between Vanilla, D-Grid, 
-----
   1.1 Quantitative Evaluation
   The following graphs were obtained when training the five_parallel_synth_data using a Vanilla Model & a D-Grid Model:
   
.. figure:: docs/train/epoch-loss_goals-directionnal.JPG
   
   As expected, the start loss for D-Grid model is lower than the Vanilla's. So, it is expected that the D-Grid Model will better predict trajectories (to be verified in 1.2)
   

   The above observation was also verified with another type of loss which is the sequential loss, as shown below:
   
.. figure:: docs/train/seq-loss_goals-directionnal.JPG

   
   1.2 Scenes Choice
   
   The following scenes show the superiority of the D-grid Model when compared to the Vanilla Model. D-Grid Model follows more closely the primary trajectory. 
   
   The best 3 scenes are: 
   
   .. figure:: docs/train/visualize_scene50932.JPG
   
   .. figure:: docs/train/visualize_scene46219.JPG
   
   .. figure:: docs/train/visualize_scene44259.JPG
   
   We played with the training options to test which training was the best one and we chose 3 more test scenes to show results.
      
   The best 3 new test scenes are:
    
   .. figure:: docs/train/visualize_avec_tout_scene53383.JPG
   This first one shows that all except the vanilla are good.
    
   .. figure:: docs/train/visualize_avec_tout_scene50782.JPG
   This one shows that the 2 differents goals directionnal models are the bests.
    
   .. figure:: docs/train/visualize_avec_tout_scene50876.JPG
   Once more goals directionalmlp and goals directional are the bests
   
   The next image is the result table, but the problem is that we can't change the table case size so the model names are not fully written in the table, so we can't know which one is directionalmlp, occupancy, or directional. But we think that the 4th one is Vanilla, the 3rd one is goals directionnal, the 2dn one is goals occupancy and the first one is goals directionnalmlp because we trained them in this order
   
   .. figure:: docs/train/Results_with_the_bug.JPG
   
   
   










































PyTorch implementation of `Human Trajectory Forecasting in Crowds: A Deep Learning Perspective <https://arxiv.org/pdf/2007.03639.pdf>`_ 

.. figure:: docs/train/cover.png

TrajNet++ is a large scale interaction-centric trajectory forecasting benchmark comprising explicit agent-agent scenarios. Our framework provides proper indexing of trajectories by defining a hierarchy of trajectory categorization. In addition, we provide an extensive evaluation system to test the gathered methods for a fair comparison. In our evaluation, we go beyond the standard distance-based metrics and introduce novel metrics that measure the capability of a model to emulate pedestrian behavior in crowds. Finally, we provide code implementations of > 10 popular human trajectory forecasting baselines.


Data Setup
==========

The detailed step-by-step procedure for setting up the TrajNet++ framework can be found `here <https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/>`_

Converting External Datasets
----------------------------

To convert external datasets into the TrajNet++ framework, refer to this `guide <https://thedebugger811.github.io/posts/2020/10/data_conversion/>`_ 

Training Models
===============

LSTM
----

The training script and its help menu:
``python -m trajnetbaselines.lstm.trainer --help``

**Run Example**

.. code-block::

   ## Our Proposed D-LSTM
   python -m trajnetbaselines.lstm.trainer --type directional --augment

   ## Social LSTM 
   python -m trajnetbaselines.lstm.trainer --type social --augment --n 16 --embedding_arch two_layer --layer_dims 1024



GAN
---

The training script and its help menu:
``python -m trajnetbaselines.sgan.trainer --help``

**Run Example**

.. code-block::

   ## Social GAN (L2 Loss + Adversarial Loss)
   python -m trajnetbaselines.sgan.trainer --type directional --augment
   
   ## Social GAN (Variety Loss only)
   python -m trajnetbaselines.sgan.trainer --type directional --augment --d_steps 0 --k 3


Evaluation
==========

The evaluation script and its help menu: ``python -m evaluator.trajnet_evaluator --help``

**Run Example**

.. code-block::

   ## TrajNet++ evaluator (saves model predictions. Useful for submission to TrajNet++ benchmark)
   python -m evaluator.trajnet_evaluator --output OUTPUT_BLOCK/trajdata/lstm_directional_None.pkl --path <path_to_test_file>
   
   ## Fast Evaluator (does not save model predictions)
   python -m evaluator.fast_evaluator --output OUTPUT_BLOCK/trajdata/lstm_directional_None.pkl --path <path_to_test_file>

More details regarding TrajNet++ evaluator are provided `here <https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/evaluator/README.rst>`_

Evaluation on datasplits is based on the following `categorization <https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/docs/train/Categorize.png>`_


Results
-------

Unimodal Comparison of interaction encoder designs on interacting trajectories of TrajNet++ real world dataset. Errors reported are ADE / FDE in meters, collisions in mean % (std. dev. %) across 5 independent runs. Our goal is to reduce collisions in model predictions without compromising distance-based metrics.

+----------------+------------+-------------------+ 
| Method         |   ADE/FDE  | Collisions        | 
+----------------+------------+-------------------+ 
| LSTM           |  0.60/1.30 | 13.6 (0.2)        | 
+----------------+------------+-------------------+ 
| S-LSTM         |  0.53/1.14 |  6.7 (0.2)        |  
+----------------+------------+-------------------+ 
| S-Attn         |  0.56/1.21 |  9.0 (0.3)        |  
+----------------+------------+-------------------+ 
| S-GAN          |  0.64/1.40 |  6.9 (0.5)        |   
+----------------+------------+-------------------+ 
| D-LSTM (ours)  |  0.56/1.22 |  **5.4** **(0.3)**| 
+----------------+------------+-------------------+ 


Interpreting Forecasting Models
===============================

+-------------------------------------------------------------------------+
|  .. figure:: docs/train/LRP.gif                                         |
|                                                                         |
|     Visualizations of the decision-making of social interaction modules |
|     using layer-wise relevance propagation (LRP). The darker the yellow |
|     circles, the more is the weight provided by the primary pedestrian  |
|     (blue) to the corresponding neighbour (yellow).                     |
+-------------------------------------------------------------------------+

Code implementation for explaining trajectory forecasting models using LRP can be found `here <https://github.com/vita-epfl/trajnetplusplusbaselines/tree/LRP>`_

Benchmarking Models
===================

We host the `Trajnet++ Challenge <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge>`_ on AICrowd allowing researchers to objectively evaluate and benchmark trajectory forecasting models on interaction-centric data. We rely on the spirit of crowdsourcing, and encourage researchers to submit their sequences to our benchmark, so the quality of trajectory forecasting models can keep increasing in tackling more challenging scenarios.

Citation
========

If you find this code useful in your research then please cite


