import math
import cv2
import pandas as pd
import numpy as np
import joblib
import warnings
import face_recognition
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

face_image = np.random.randint(256, size=(16, 16))
print(face_image, end=" = face_image before\n\n")

for i in range(0, 16):
    for j in range(0, 16):
        face_image[i, j] = int(math.floor(face_image[i][j] / 32))

print(face_image, end=" = face_image after\n\n")

glcm = graycomatrix(face_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True,
                            normed=True)
print(glcm, end=" = glcm\n\n")

props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
print(glcm_props, end=" = glcm_props\n\n")
