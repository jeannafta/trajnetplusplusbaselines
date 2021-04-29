
TrajNet++ : The Trajectory Forecasting Framework
================================================

Milestone 1
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
   
   As for the results of each model: (from top to bottom: MLP, Occasional, D-Grid, Vanilla)
   
   
   
   ADD DISCUSSION
   
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


