import math
import cv2
import pandas as pd
import numpy as np
import joblib
import warnings
import face_recognition
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = [get_pixel(img, center, x - 1, y - 1), get_pixel(img, center, x - 1, y),
              get_pixel(img, center, x - 1, y + 1), get_pixel(img, center, x, y + 1),
              get_pixel(img, center, x + 1, y + 1), get_pixel(img, center, x + 1, y),
              get_pixel(img, center, x + 1, y - 1), get_pixel(img, center, x, y - 1)]

    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def calc_glcm_all_agls(img, props):
    glcm = graycomatrix(img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=8, symmetric=True,
                        normed=True)

    print(glcm, end=" = glcm\n\n")

    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)

    return feature


if __name__ == '__main__':
    # inisialisasi
    data_path = "C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\10_random_lbp\\glcm24.csv"
    image_path = "C:\\Users\\Marcel\\Downloads\\face_anti-spoofing\\face_anti-spoofing\\imgs_validation\\real\\" \
                 "30-1-image1-test.jpeg"
    model_path = "C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\model\\lbp_model1.pkl"

    video_capture = cv2.VideoCapture(0)  # webcamera

    if not video_capture.isOpened():
        print("Unable to access the camera")
    else:
        print("Access to the camera was successfully obtained")
    print("Streaming started")

    while True:
        ret, frame = video_capture.read()

        df = pd.read_csv(data_path)
        df = df.drop('Unnamed: 0', axis=1)
        X = df.drop('label', axis=1)

        scaler = StandardScaler().fit(X)
        model = joblib.load(model_path)

        # image preparation
        image = frame
        result_image = image

        # face detection
        face_locations = face_recognition.face_locations(image)
        print(face_locations, end=" = face_location\n\n")

        if face_locations: # jika terdeteksi wajah
            for top, right, bottom, left in face_locations:
                face_image = image[top:bottom, left:right]
                cv2.rectangle(result_image, (left, top), (right, bottom), (255, 0, 0), 2)
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                face_image = cv2.resize(face_image, (256, 256))

                print(face_image, end=" = face_image\n\n")

                # lbp
                lbp_image = np.zeros((256, 256), np.uint8)
                for l in range(0, 256):
                    for m in range(0, 256):
                        lbp_image[l, m] = lbp_calculated_pixel(face_image, l, m)

                print(lbp_image, end=" = lbp_image\n\n")

                # glcm
                glcm = np.zeros((8, 8), np.uint8)
                glcm_image = np.zeros((256, 256), np.uint8)
                properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
                glcm_all_agls = []

                for i in range(0, 256):
                    for j in range(0, 256):
                        glcm_image[i, j] = int(math.floor(lbp_image[i][j] / 32))

                print(glcm_image, end=" = glcm_image\n\n")
                glcm_all_agls.append(calc_glcm_all_agls(glcm_image, props=properties))
                print(glcm_all_agls, end=" = glcm_all_agls\n\n")

                # standarisasi nilai-nilai dari dataset
                glcm_all_agls = scaler.transform(glcm_all_agls)
                print(glcm_all_agls, end=" = glcm_all_agls\n\n")

                # predict
                predict = model.predict(glcm_all_agls)[0]
                print(predict)

        else:
            print("tidak ada wajah")

        cv2.imshow("Space to quit", result_image)

        key = cv2.waitKey(1)
        if key % 256 == 32:  # Space code
            break

    # When everything done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    print("Streaming ended")
