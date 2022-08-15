# Drivable Area Segmentation
Segment the Drivable area for a vehicle


The data for training was obtained from the [Berkeley Deep Drive Dataset](https://bdd-data.berkeley.edu). Only 3000 images were used for training, validation, and testing. With such a limited dataset, a pretrainined DeepLabV3 with a ResNet50 backbone was used. The data was augmented with random horizontal and vertical flips, color jitter, and random shadows.

Experiments revealed that the only fine tuning the classifier portion of the model led to inferior results when compared to fine tuning the entire model. The model was able to perform well on the data and even show decent generaliztion after a single epoch. 

Experiments with Categorical Cross Entropy (CCE) and Dice loss functions were performed to see which one (or combination of) could lead to better results. Using the Dice loss alone did not lead to a well generalizing model, the CCE alone and an equal weighting of CCE and Dice losses lead to similar results. The performance of both classes of models is satisfactory, further training on a large diverse dataset with more intense augmentation may lead to better generalization.


Even though the models were trained and validated on Berkeley Deep Drive Dataset dataset, the real test for generalization was on out of sample videos. Some of the results are shown below


#### A basic example from the KITTI dataset
https://user-images.githubusercontent.com/60835780/184648802-954c6357-5427-43a4-ab9e-299dadc1d0a8.mp4




#### A scenario in Paris where there are no lane line
https://user-images.githubusercontent.com/60835780/184653271-777aefe6-5a84-467d-a382-924c0686e175.mp4






