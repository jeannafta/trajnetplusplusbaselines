
TrajNet++ : The Trajectory Forecasting Framework
================================================

Milestone 1 - Setting Up TrajNet++ and training a D-LSTM model
==========

+-----------------------------+
| **Group A**                 | 
+-----------------------------+ 
| Gaelle Abi Younes           |
+-----------------------------+
| Jean Naftalski              |  
+-----------------------------+ 
| Florent Zolliker            |  
+-----------------------------+ 

AICrowd Team Name: jeannafta

1. Comparaison between Vanilla, D-Grid, D-Grid MLP, D-Grid Occupancy on five_parallel_synth_data
-----

   **1.1 Quantitative Evaluation**
   
   The following graphs were obtained when training the five_parallel_synth_data using a Vanilla Model & a D-Grid Model:
   
.. figure:: docs/train/epoch-loss_goals-directionnal.JPG
   
   As expected, the start loss for D-Grid model is lower than the Vanilla's. So, it is expected that the D-Grid Model will better predict trajectories (to be verified in 1.2)
   

   The above observation was also verified with another type of loss which is the sequential loss, as shown below:
   
.. figure:: docs/train/seq-loss_goals-directionnal.JPG

   
   **1.2 Scenes Choice**
   
   The following scenes show the superiority of the D-grid Model when compared to the Vanilla Model. D-Grid Model follows more closely the primary trajectory. 
   
   The best 3 scenes are: 
   
   .. figure:: docs/train/visualize_scene50932.JPG
   
   .. figure:: docs/train/visualize_scene46219.JPG
   
   .. figure:: docs/train/visualize_scene44259.JPG
   
   We played with the training options to test whiche training was the best one and we chose the following 3 scenes:
    
   .. figure:: docs/train/visualize_avec_tout_scene53383.JPG
   
   This scene shows the superiority of all models over the vanilla model.  
    
   .. figure:: docs/train/visualize_avec_tout_scene50782.JPG
   
   This scene demonstrates the superiority of directionnal model.
    
   .. figure:: docs/train/visualize_avec_tout_scene50876.JPG
   
   However, this schene showed that in some cases, the directional MLP model behaves better than the directional LSTM. 
   
   As for the results of each model: (from top to bottom: Directionalmlp, Occupancy, Directional, Vanilla)
   
   .. figure:: docs/train/Results_new.JPG 
   
   labels:
   
    The average displacement error (ADE) is the average distance between the ground truth and our prediction over all predicted time steps.
    
    The final displacement error (FDE) is the distance between the predicted final destination and the true final destination at the end of the prediction period.
    
   We can see that the directionnal have lower values for ADE and FDE than other models, this means that it must be usually the best one to approximate the groundtruth path. 
   
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

It seems that the legend is wrong because the Vanilla model is always better than our submitted model, on the wo images even with our Vanilla training as submission the Vanilla has lower ADE, FDE GT Collision and Pred. Collision. We think that the legend should be inversed, because our trained model should be better than the Vanilla basic one.
It also seems that the NLL is not working by now because the value is 0 in both case for our submitted model.

Considering this, we can see interesant things:
  The Pred. Collision and the GT Collision is lower for our trained model than for the Vanilla one, this was expected.
  
  We can see that ADE and FDE are only a little bit lowe, this is not that good, it means that the submitted model is not that much better than the Vanilla one considering trajectory predictions.
  

If we look the exemple we saw on the course the legend was correct and the tested model is much better than the Vanilla one in every sections except GT Collision:

.. figure:: docs/train/UNIMODAL_MULTIMODAL_ex_du_cours.JPG


3. Retraining using CFF datas
-----
We tried to use all the data set including cff datas to train our model, the induced model is not as good as before. The trained model without cff data is better. There can be multiple reasons to this, but the main one is that cff data were too noisy and so it's lowering the model training performances. The final difference between the two models is not that high because having more data is a good thing overall, so it lowers the bad impact of the noisy datas.

================================================

Milestone 2 - Implementing Social Contrastive Learning
==========

1. Introduction
-----

**1.1 Problem Statement**
So far, the trained model is not socially aware, meaning that it is not able to differentiate between socially acceptable behaviors and what is not. However, how can the model differentiate between the two and avoid socially unfavorable events such as collisions, when these scenarios rarely happen in real life and are almost completey absent in real data? 
Based on this idea, the concept of social contrastive learning was created, and will be implemented as part of this milestone. 

**1.2 What is Social Contrastive Learning?**

The key behind implementing contrastive learning is data augmentation. The type of data that needs to be created is "dangerous" data that will allow the model to become more socially aware. This data, also called Negative Data, is generated at a certain time for all neighbors of a scene and that using their trajectory and position. while Positive Data corresponds to the groundtruth position of the primary agent at that same time. 
The model should then be able to correctly predict the trajectory of the primary agent while avoiding these unfavorable events. 
The advantage of this method is that it introduces a social contrastive loss that encourages the encoded motion representation to preserve sufficient information for distinguishing a positive future event from a set of negative ones (Liu, Y., et al.) https://arxiv.org/pdf/2012.11717.pdf 

   
    ..figure:: docs/train/contrastive_learning_representation.JPG

2. Implementation
-----
**2.1 Sampling strategies**

Eventhough several sampling strategies exist, only two were implemented within the scope of this milestone: 
     
     
     2.1.1 Spatial sampling
     
This method consists in drawing negative samples based on locations of neighbouring agents at a fixed time step. From this position, 8 more positions are generated in such a way to form a circle around the actual position. In total, 9 negative samples are generated per agent and some noise was also added to leave some room for error. One of the many challenges encountered to accomplish this task was the variability of neighbors in each scene. To deal with that, a NaN tensor was created having of its dimension equal to the maximal number of neighbors in that particular batch, and another of its dimensions equal to the number of scenes in the batch (1 batch contains 8 scenes). Negative samples were then generated and replaced the NaN values when possible. However, some NaN values were still present in the negative samples when the number of neighbors in that scene is less than the maximum number of neighbors. Once the negative data generated, some values were considered easy if they were too far from the primary agent and too hard if they were too close. If the distance between the agent of interest and its neighbors i.e., distance between negative and positive data was smaller than a minimum separation and larger than a maximum separation, the coordiantes of these specific locations were set to NaN. Another source of NaN values is missing values from the data itself. 
The NaN values were then replaced by -10 meaning that this agent is far from the primary agent and therefore is not of interest. 
Another crucial step of that process, was to decide on a step time within the sampling horizon. For a sampling horizon equal to 4, the time step before the last i.e. t=3 was  "yields significant performance gains on both reward and collision metrics" (Liu, Y., et al.) https://arxiv.org/pdf/2012.11717.pdf. 
Positive samples correspond to the groundthruth of primary agent at a specific time with some noise added to it. 
     
      2.1.2 Event sampling
     
The third sampling method consists in drawing negative samples based on regions of other agents across multiple time steps. This means that it is close to the Social sampling but multiple time steps are considered, meaning the entire sampling horizon. 
   
   
**2.2 Query**
   
To accurately predict the trajectory of the primary agent, some important features need to be learnt from the history of the primary agent. A batch feat was generated from 9 previous observations. A 2 layer MLP are added downstream because the last layer is too specific to the pretrained task which drives the model to underperform.
   

**2.3 Embedding**
Once the query, positive and negative data were obtained, they were embedded in the space and normalized across the features dimension. 


**2.4 Similarity**
This task is established in order to maximize similarity between the extracted motion representation and the representation of positive events, and minimize similarity between the extracted motion representation and the representation of negative events. 

**2.5 Loss**
Loss is computed between the logits and labels. Labels were drawn from the data itself (Self-supervised Learning). An NCE Loss is generated then given a certain weight λ (hyperparameter to be fine-tuned while training) and then added to the basic loss. 

**2.6 Settings & Training**
Given 9 time steps of observations as imput, we want to predict future trajectories for 12 time steps for the primary agent.
As in milestone 1, we will compare the models performances with reference to FDE (Final Displacement Error) and COL-1 (collision rate).

All models will be trained using Adam optimizer.

Since the D-Grid model yields better results, as shown in Milestone 1, it will be used to train models in this Milestone. 


3. Results and Hyperparameter Fine-Tuning
-----
Trained models on synth_data and real_data were evaluated and submitted on the AICrowd Platform (https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge/leaderboards)

**Attempt 1:**

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
       

  
