# # example of calculating the frechet inception distance
# import numpy
# from numpy import cov
# from numpy import trace
# from numpy import iscomplexobj
# from numpy.random import random
# from scipy.linalg import sqrtm
 
# # calculate frechet inception distance
# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = numpy.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if iscomplexobj(covmean):
#         covmean = covmean.real
#         # calculate score
#         fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
#         return fid
 
# # define two collections of activations
# act1 = random(10*2048)
# act1 = act1.reshape((10,2048))
# act2 = random(10*2048)
# act2 = act2.reshape((10,2048))
# # fid between act1 and act1
# fid = calculate_fid(act1, act1)
# print('FID (same): %.3f' % fid)
# # fid between act1 and act2
# fid = calculate_fid(act1, act2)
# print('FID (different): %.3f' % fid)



import numpy as np
from einops import rearrange 
# import tensorflow as tf
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from tensorflow.keras.preprocessing import image
from numpy import cov, trace, iscomplexobj
from scipy.linalg import sqrtm
from PIL import Image 
import torch
from torchvision import datasets, transforms
import cv2
import os

model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
model.eval()
preprocess = transforms.Compose([
    transforms.Resize([299, 299])
])
images1_path = 'Datasets/alphanumeric_dataset/alphanumeric_dataset/Train'
images2_path = 'static/Predicted_Alphanumeric_Image/'

def load_images_from_folder(folder):
    images1 = []
    images2 = []
    for filename in os.listdir(folder):
        folder_name = filename.split('_')[0]
        images1_path = 'Datasets/alphanumeric_dataset/alphanumeric_dataset/Train/'+folder_name
        for images in os.listdir(images1_path):
            img1 = cv2.imread(os.path.join(images1_path,images))
            images1.append(img1)
            break
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images2.append(img)
    return images1, images2

images1, images2 = load_images_from_folder(images2_path)

images1 = [preprocess(rearrange(torch.from_numpy(item).float(), 'h w c -> c h w')) for item in images1]
images2 = [preprocess(rearrange(torch.from_numpy(item).float(), 'h w c -> c h w')) for item in images2]

images1 = torch.stack(images1)
images2 = torch.stack(images2)

# input_tensor1 = preprocess(input_image1)
# input_tensor2 = preprocess(input_image2)
# input_batch1 = input_tensor1.unsqueeze(0) # create a mini-batch as expected by the model
# input_batch2 = input_tensor2.unsqueeze(0)
# # move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch1 = images1.to('cuda')
    input_batch2 = images2.to('cuda')
    model.to('cuda')
# input_batch1 = images1
# input_batch2 = images2
with torch.no_grad():
  output1 = model(input_batch1)
  output2 = model(input_batch2)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes

# Calculate Frechet Inception Distance (FID)
def calculate_fid(output1, output2):
    # Extract activations from images
    # act1 = extract_activations(images1)
    # act2 = extract_activations(images2)    
    # Calculate mean and covariance statistics
    mu1, sigma1 = output1.mean(axis=0), cov(output1.detach().cpu().numpy(), rowvar=False)
    mu2, sigma2 = output2.mean(axis=0), cov(output2.detach().cpu().numpy(), rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2)**2.0).detach().cpu().numpy()
    
    # Calculate sqrt of product between covariances
    covmean = np.sqrt(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate Frechet Inception Distance
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Load your images (e.g., using PIL.Image or TensorFlow's ImageDataGenerator)


# Calculate FID between two sets of images
fid = calculate_fid(output1, output2)
print('FID between images1 and images2: %.3f' % fid)