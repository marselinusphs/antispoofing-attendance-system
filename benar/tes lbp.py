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


face_image = np.random.randint(8, size=(8, 8))
print(face_image)
x, y = 2, 2
print(face_image[x-1][y-1])
print(face_image[x-1][y])
print(face_image[x-1][y+1])
print(face_image[x][y+1])
print(face_image[x+1][y+1])
print(face_image[x+1][y])
print(face_image[x+1][y-1])
print(face_image[x][y-1])
