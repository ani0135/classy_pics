############### AI Innovation 3.0 Hackathon by Yamaha and IIT Mandi ####


We have 2 datasets in Folder "Datasets"
    1. alphanuemric_dataset/alphanumeric_dataset
        ---- Train
        ---- Validation
    2. Animals10
        ---- raw-img

You need to create directories like above for training.

Model is built using Python 3.8.5
To install correct packages, run <font color="green"> pip install requirement.txt </font>

To train the model on alphanumeric datasets run following command
 " python class_alphanumeric.py "


To train the model on Animal datasets run following command
 " python class_animal.py "

weight files are stored in Checkpoint folder

To test the model on alphanumeric and animal both datasets run following command
 " python class_alpha_animal_test_with_flask.py "

it also serves model output (image) to website using flask, currently port is 5001.
For the front-end, Website is in "Hackathon_Web_Page" folder.


To test the model individually for each datasets, we created two files -->
class_alphanumeric_test.py
class_animal_test.py 

predicted images are stored in " static/Predicted_Animal_Image/ " for Animal, " static/Predicted_Alphanumeric_Image/ " for alphanumeric datasets.

==========================
For Metrics Evaluation, we created two scores, FID and IS score.
to run use fdi_score/fdi_score.py file and IS_Score.py file on predicted image
