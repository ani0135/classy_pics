import os
import cv2
import numpy as np
from skimage.transform import resize
from numpy import expand_dims, log, mean, std, exp
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import torchvision.transforms as transforms

def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)

def calculate_inception_score(images, n_split=5, eps=1E-16):
    model = InceptionV3()
    scores = list()
    n_part = len(images) // n_split
    for i in range(n_split):
        ix_start, ix_end = i * n_part, (i+1) * n_part
        subset = images[ix_start:ix_end]
        subset = scale_images(subset, (299, 299, 3))
        subset = preprocess_input(subset)
        p_yx = model.predict(subset)
        p_y = expand_dims(p_yx.mean(axis=0), 0)
        kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        avg_kl_d = mean(sum_kl_d)
        is_score = exp(avg_kl_d)
        scores.append(is_score)
        print("Iteration ",i," : ", is_score)
    is_avg, is_std = mean(scores), std(scores)
    return is_avg, is_std

input_folder = "Predicted_Alphanumeric_Image"
items = os.listdir(input_folder)

images = []

for item in items:
    img = cv2.imread(os.path.join(input_folder, item))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299, 299))
    images.append(img)

images = np.array(images)

is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)
