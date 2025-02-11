TrajNet++ : The Trajectory Forecasting Framework
================================================

+-----------------------------+
| **Group A**                 | 
+-----------------------------+ 
| Gaelle Abi Younes           |
+-----------------------------+
| Jean Naftalski              |  
+-----------------------------+ 
| Florent Zolliker            |  
+-----------------------------+ 

AICrowd Team Name: CIVIL-459_GroupA

Milestone 1 - Setting Up TrajNet++ and training a D-LSTM model
==========


1. Comparaison between Vanilla, D-Grid, D-Grid MLP, D-Grid Occupancy on five_parallel_synth_data
-----

   **1.1 Quantitative Evaluation**
   
   The following graphs were obtained when training the five_parallel_synth_data using a Vanilla Model & a D-Grid Model:
   
.. figure:: docs/train/epoch-loss_goals-directionnal.JPG
   
   As expected, the start loss for D-Grid model is lower than the Vanilla's. So, it is expected that the D-Grid Model will better predict trajectories (to be verified in 1.2).
   

   The above observation was also verified with another type of loss which is the sequential loss, as shown below:
   
.. figure:: docs/train/seq-loss_goals-directionnal.JPG

   
   **1.2 Scenes Choice**
   
   The following scenes show the superiority of the D-grid Model when compared to the Vanilla Model. D-Grid Model follows more closely the primary trajectory. 
   
   The best 3 scenes are: 
   
   .. figure:: docs/train/visualize_scene50932.JPG
   
   .. figure:: docs/train/visualize_scene46219.JPG
   
   .. figure:: docs/train/visualize_scene44259.JPG
   
   We played with different interaction modules to identify the most performing one. The following scenes were chosen:
    
   .. figure:: docs/train/visualize_avec_tout_scene53383.JPG
   
   This scene shows the superiority of all interaction models over the vanilla model.  
    
   .. figure:: docs/train/visualize_avec_tout_scene50782.JPG
   
   This scene demonstrates the superiority of directionnal model.
    
   .. figure:: docs/train/visualize_avec_tout_scene50876.JPG
   
   However, this schene showed that in some cases, the directional MLP model behaves better than the directional LSTM. 
   
   As for the results of each model:
   
   .. figure:: docs/train/Results_new.JPG 
   
   labels:
   
    The average displacement error (ADE) is the average distance between the ground truth and our prediction over all predicted time steps.
    
    The final displacement error (FDE) is the distance between the predicted final destination and the true final destination at the end of the prediction period.
    
   Using the directionnal grid interaction module, the lowest values of ADE and FDE were obtained when compared to all others models. This interaction module will be retained for the upcoming steps of the project. 
   
2. Comparaison between Vanilla & D-Grid Model using synth_data & real_data
-----

After evaluating the Vanilla Model on AICrowd, the following results for the different losses were obtained:

.. figure:: docs/train/vanilla.png

After evaluating the D-Grid Model on AICrowd, the following results for the different losses were obtained:

.. figure:: docs/train/directional.png

We can look at theses two images and see some terms evaluating the models.
  In the left circles (unimodal) we have:
    The average displacement error (ADE) is the average distance between the ground truth and our prediction over all predicted time steps.

    The final displacement error (FDE) is the distance between the predicted final destination and the true final destination at the end of the prediction period.
  
    The groundtruth collision (GT Collision) is the percentage of collision of primary pedestrian with the neighbors in the groundtruth future scene.
  
    The prediction Collision (Pred. Collision) is the percentage of collision of primary pedestrian with the neighbors in the predicted future scene.
  
  In the right circle (multimodal) we have:
    The NLL. Given multiple samples, the metric calculates the average negative log-likelihood of groundtruth trajectory over the prediction horizion.
    
    The top3_ADE. Given 3 output predictions, the metrics calculate the ADE of the prediction closest to the groundtruth trajectory.
    
    The top3_FDE. Given 3 output predictions, the metrics calculate the FDE of the prediction closest to the groundtruth trajectory.


  It was observed that Pred. Collision and GT Collision are lower for the trained models considering interactions than for the Vanilla one. This is explained by the fact that the vanilla model does not take into account intercations between the different pedestrians of the scenes which will result in less accurate predictions.

.. figure:: docs/train/UNIMODAL_MULTIMODAL_ex_du_cours.JPG


3. Training using CFF datas
-----
We tried to use all the data set including cff datas to train our model, the obtained model did not perform as good as the previous model. TThis is mainly explained by the fact that CFF data are noisy, which drives the model to underperform. The final difference between the two models is not that high because having more data is a good thing overall, so it lowers the bad impact of the noisy data.

================================================

Milestone 2 - Implementing Social Contrastive Learning
==========

1. Introduction
-----

**1.1 Problem Statement**

So far, the trained model is not socially aware, meaning that it is not able to differentiate between socially acceptable behaviors and what is not. However, how can the model differentiate between the two and avoid socially unfavorable events such as collisions, when these scenarios rarely happen in real life and are almost completey absent in real data? 
Based on this idea, the concept of social contrastive learning was created, and will be implemented as part of this milestone. 

**1.2 What is Social Contrastive Learning?**

The key behind implementing contrastive learning is data augmentation. The type of data that needs to be created is "dangerous" data that will allow the model to become more socially aware. This data, also called Negative Data, is generated at a certain time for all neighbors of a scene and that using their trajectory and position. While Positive Data corresponds to the groundtruth position of the primary agent at that same time. 
The model should then be able to correctly predict the trajectory of the primary agent while avoiding these unfavorable events. 
The advantage of this method is that it introduces a social contrastive loss that encourages the encoded motion representation to preserve sufficient information for distinguishing a positive future event from a set of negative ones `(Liu, Y., et al.) <https://arxiv.org/pdf/2012.11717.pdf>`_

.. figure:: docs/train/contrastive_learning_representation.JPG

2. Implementation
-----
**2.1 Sampling strategies**

Eventhough several sampling strategies exist, only two were implemented within the scope of this milestone: 
     
     
     2.1.1 Spatial sampling
     
This method consists in drawing negative samples based on locations of neighbouring agents at a fixed time step. From this position, 8 more positions are generated in such a way to form a circle around the actual position. In total, 9 negative samples are generated per agent and some noise was also added to leave some room for error. One of the many challenges encountered to accomplish this task was the variability of neighbors in each scene. To deal with that, a NaN tensor was created having one of its dimensions equal to the maximum number of neighbors in that particular batch, and another of its dimensions equal to the number of scenes in the batch (1 batch contains 8 scenes). Negative samples were then generated and replaced the NaN values when possible. However, some NaN values were still present in the negative samples when the number of neighbors in that scene is less than the maximum number of neighbors. Once the negative data generated, some values were considered easy if they were too far from the primary agent and too hard if they were too close. If the distance between the agent of interest and its neighbors i.e., distance between negative and positive data was smaller than a minimum separation and larger than a maximum separation, the coordinates of these specific locations were set to NaN. Another source of NaN values is missing values from the data itself. 
The NaN values were then replaced by -10 meaning that this agent is far from the primary agent and therefore is not of interest. 
Another crucial step of that process, was to decide on a step time within the sampling horizon. For a sampling horizon equal to 4, the time step before the last i.e. t=3  "yields significant performance gains on both reward and collision metrics" `(Liu, Y., et al.) <https://arxiv.org/pdf/2012.11717.pdf>`_ .
Positive samples correspond to the groundthruth of primary agent at a specific time with some noise added to it. 
     
  Negative and positive data were visualized:
  
  In this scene, no NaN values were encountered. 

.. figure:: docs/train/scene_and_samples0.png

Samples with NaN data were replaced with -10 as can be seen in Figure below:

.. figure:: docs/train/scene_and_samples1.png    
      
      2.1.2 Event sampling
     
This sampling method consists in drawing negative samples based on regions of other agents across multiple time steps. This means that it is close to the Social sampling but multiple time steps are considered, meaning the entire sampling horizon steps. 
   
**2.2 Query**
   
To accurately predict the trajectory of the primary agent, some important features need to be learnt from the history of the primary agent. A batch feat was generated from 9 previous observations. Here we have chosen to keep only the first prediction (prediction at time 0 of the batch feat), for the calculation of the query, but this will be improved in milestone 3. 
   
.. figure:: docs/train/Time.png

**2.3 Embedding**

A 2 layer MLP (Projection Head) was used to encode the history of observations into an 8-dimensional  embedding  vector. Positive and negative samples were also embedded in the same space using spatial or event encoder. Then the embedded vectors were normalized across the features dimension. 


**2.4 Similarity**

This task is established in order to maximize similarity between the extracted motion representation and the representation of positive events, and minimize similarity between the extracted motion representation and the representation of negative events. 

**2.5 Loss**

Loss is computed between the logits and labels. An NCE Loss is then generated given a certain weight λ (hyperparameter to be fine-tuned while training) and then added to the basic loss. 

**2.6 Settings & Training**

Given 9 time steps of observations as input, we want to predict future trajectories for 12 time steps for the primary agent.
As in milestone 1, we will compare the models performances with reference to FDE (Final Displacement Error) and COL-1 (collision rate).

All models will be trained using Adam optimizer.

Since the D-Grid model yields better results, as shown in Milestone 1, it will be used to train models in this Milestone. 


3. Results and Hyperparameter Fine-Tuning
-----
Trained models on synth_data and real_data were evaluated and submitted on the `AICrowd Platform <https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/leaderboards>`_

**Best Attempt :**

+-----------------------------+-----------------------------+
| **Hyperparamter**           |        **Value**            |
+-----------------------------+-----------------------------+ 
| Learning Rate               |           0.001             |
+-----------------------------+-----------------------------+
| Contrast Sampling           |           Multi             |
+-----------------------------+-----------------------------+ 
| λ                           |            0.1              |
+-----------------------------+-----------------------------+
| Epochs                      |            16               |
+-----------------------------+-----------------------------+
| Temperature                 |            0.07             |
+-----------------------------+-----------------------------+   
       
Obtained results:

* FDE: 1.190

* COL-I: 4.830

.. figure:: docs/train/summary.png

================================================

Milestone 3 - 
==========

In this milestone, we played with the parameters in order to boost the performance of the model:

+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|**Used milestone 1 ?**|**Epochs**|**Contrast weight λ**|**Learning Rate**|**Contrast Sampling**|**Temperature**|**Horizon**|**Noise augmentation ?**|**FDE**|**COL-I**|
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |          0.1        |       1e-3      |         Multi       |    0.07       |      4    |           NO           |  1.330|  5.730  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |          4.0        |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.350|  6.260  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |         8.0         |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.240|  5.130  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |         9.0         |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.240|  5.970  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |          10.0       |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.210|  6.150  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |          11.0       |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.330|  6.090  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |          12.0       |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.300|  5.250  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    5     |          15.0       |       5e-4      |         Multi       |      0.1      |      4    |           YES          |  1.230|  5.970  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+

Adding noise in addition to contrast weights of 0.1 & 10 were found to be useful to improve the performance of the model. 


**SGAN:**

In this section, the sgan baseline was implemented in order to output multiple predictions (multimodality concept). 
The following parameters were used: 

* Learning rate = 1e-3
* Epochs: 25
* With Discriminator

The following results were obtained:

* FDE = 1.24
* Col-1 = 5.61

The trained model performed less better than the SocialNCE model. In case hyperparameters were changed, better results can be expected. We tried adding social loss to the SGAN Model however, it took very much to train. Therefore the code needs to be more optimized (future work). 

**IMPROVEMENT OF EVENT SAMPLING:**

For this third milestone we wanted to improve our implementation of event sampling (see part 2.1.2). In our previous implementation, we calculated the query at time 0 of the prediction in batch feat, and we computed the similarity with the positive and negative samples generated at time 0, 1, ..., horizon (in the future). 
Now we would like to give some additional information to calculate the loss more accurately. For this purpose, we decided to calculate the queries at each time t between 0 and pred_length-horizon, comparing them to the negative and positive samples generated at time t, t+1, ..., t+horizon-1 (see diagram below for the arrangement when calculating the similarities). With our parameters (horizon=4 and pred_lenth=12) we get 9 times more logits and therefore we gave more information about future good/bad events that could happen to the primary neighbor. 

.. figure:: docs/train/NEW.jpg

This approach has allowed us to improve our results:

+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|**Used milestone 1 ?**|**Epochs**|**Contrast weight λ**|**Learning Rate**|**Contrast Sampling**|**Temperature**|**Horizon**|**Noise augmentation ?**|**FDE**|**COL-I**|
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    15    |          0.1        |       1e-3      |         Multi       |      0.1      |      4    |           YES          |  1.180|  5.790  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+
|     YES-25epochs     |    15    |          10.0       |       1e-3      |         Multi       |      0.1      |      4    |           YES          |  1.210|  4.710  |
+----------------------+----------+---------------------+-----------------+---------------------+---------------+-----------+------------------------+-------+---------+

With λ = 0.1, the smallest FDE of 1.18 was obtained and,

using λ = 10, the smallest COL-1 of 4.71 was obtained

At the end, our best obtained model used interaction modules and contrastive learning. The method implemented in milestone 3 combined with event sampling have given the best results. 
