import math
import cv2
import pandas as pd
import numpy as np
import joblib
import face_recognition
import warnings
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


class Model:
    def __init__(self, data_path, model_path):
        self.predict = None
        self.scaler = None
        self.image_glcm = None
        self.glcm_all_agls = None
        self.image_to_process = None
        self.face_locations = None
        self.image_to_show = None
        self.image_lbp = None
        self.data_path = data_path
        self.model_path = model_path

        self.df = pd.read_csv(self.data_path)
        self.df = self.df.drop('Unnamed: 0', axis=1)
        self.X = self.df.drop('label', axis=1)
        self.scaler = StandardScaler().fit(self.X)
        self.model = joblib.load(self.model_path)

    def __get_pixel(self, img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def __lbp_calculated_pixel(self, img, x, y):
        center = img[x][y]
        val_ar = [self.__get_pixel(img, center, x - 1, y - 1), self.__get_pixel(img, center, x - 1, y),
                  self.__get_pixel(img, center, x - 1, y + 1), self.__get_pixel(img, center, x, y + 1),
                  self.__get_pixel(img, center, x + 1, y + 1), self.__get_pixel(img, center, x + 1, y),
                  self.__get_pixel(img, center, x + 1, y - 1), self.__get_pixel(img, center, x, y - 1)]

        power_val = [128, 64, 32, 16, 8, 4, 2, 1]
        val = 0

        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]

        return val

    def lbp_preprocessing(self):

        self.image_to_process = cv2.cvtColor(self.image_to_process, cv2.COLOR_BGR2GRAY)
        self.image_lbp = np.zeros((256, 256), np.uint8)

        for l in range(0, 256):
            for m in range(0, 256):
                self.image_lbp[l, m] = self.__lbp_calculated_pixel(self.image_to_process, l, m)

    def calc_glcm_all_agls(self, img, props):
        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=8,
                            symmetric=True, normed=True)

        feature = []
        glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
        for item in glcm_props:
            feature.append(item)

        return feature

    def glcm_preprocessing(self):
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        self.glcm_all_agls = []

        self.image_glcm = np.zeros((256, 256), np.uint8)
        for i in range(0, 256):
            for j in range(0, 256):
                self.image_glcm[i, j] = int(math.floor(self.image_lbp[i][j] / 32))

        self.glcm_all_agls.append(self.calc_glcm_all_agls(self.image_glcm, props=properties))

    def start_predict(self):
        video_capture = cv2.VideoCapture(0)  # webcamera

        if not video_capture.isOpened():
            print("Unable to access the camera")
        else:
            print("Access to the camera was successfully obtained")
        print("Streaming started")

        while True:
            ret, frame = video_capture.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            self.image_to_show = frame
            self.face_locations = face_recognition.face_locations(frame)

            if self.face_locations:
                for top, right, bottom, left in self.face_locations:
                    self.image_to_process = frame[top:bottom, left:right]
                    self.image_to_process = cv2.resize(self.image_to_process, (256, 256))

                    self.lbp_preprocessing()
                    self.glcm_preprocessing()

                    # standarisasi nilai-nilai dari dataset
                    self.glcm_all_agls = self.scaler.transform(self.glcm_all_agls)
                    self.predict = self.model.predict(self.glcm_all_agls)[0]

                    if self.predict == "real":
                        cv2.rectangle(self.image_to_show, (left, top), (right, bottom), (255, 0, 0), 2)
                        cv2.rectangle(self.image_to_show, (left, top - 40), (right, top), (255, 0, 0), cv2.FILLED)
                    else:
                        cv2.rectangle(self.image_to_show, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(self.image_to_show, (left, top - 40), (right, top), (0, 0, 255), cv2.FILLED)

                    # cv2.putText(self.image_to_show,
                    #             f"{self.predict} ({math.floor(max(self.model.predict_proba(self.glcm_all_agls)[0]) * 100)}%)",
                    #             (left + 10, top - 10), cv2.FONT_HERSHEY_PLAIN, 2.0, (247, 247, 247), 2)

                    cv2.putText(self.image_to_show, f"{self.predict}",
                                (left + 10, top - 10), cv2.FONT_HERSHEY_PLAIN, 2.0, (247, 247, 247), 2)
            else:
                print("Wajah tidak ditemukan")

            # Display the resulting frame
            cv2.imshow("Spoofing detector - press Space to quit", self.image_to_show)

            # Exit with Space
            key = cv2.waitKey(1)
            if key % 256 == 32:  # Space code
                break

        # When everything done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
        print("Streaming ended")


if __name__ == '__main__':
    model1 = Model(
        "../sistempresensi/static/lbp_1_glcm24.csv",
        "../sistempresensi/static/model_terbaik.pkl"
    )

    model1.start_predict()
