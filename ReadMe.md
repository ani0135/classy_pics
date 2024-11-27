# AI Innovation 3.0 Hackathon by Yamaha and IIT Mandi

This repository contains the implementation for training and testing classification models on two datasets: **Alphanumeric** and **Animals10**. Additionally, it includes a Flask-based web interface for visualizing predictions.

---

## **Datasets**

### 1. **Alphanumeric Dataset**
Path: `alphanuemric_dataset/alphanumeric_dataset`  
- Contains two subdirectories:
  - `Train`
  - `Validation`

### 2. **Animals10 Dataset**
Path: `Animals10/raw-img`  
- Contains raw images for classification.

Ensure datasets are organized in the same structure before proceeding with training or testing.

---

## **Setup Instructions**

1. **Environment**  
   - Python version: **3.8.5**
   - Install the required packages by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Directory Structure**  
   Create directories as described above for seamless training and testing.

---

## **Training**

### Alphanumeric Dataset
Run the following command to train the model on the **Alphanumeric Dataset**:
```bash
python class_alphanumeric.py



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
