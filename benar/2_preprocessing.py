# PREPROCESSING + MODELING

import os
import math
import cv2
import pandas as pd
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from glob import glob


class Model:
    def __init__(self, nama, source_path, img_size, save_path):
        self.nama = nama
        self.source_path = source_path
        self.img_size = img_size
        self.save_path = save_path

    def __save_frame(self, image, save_path):
        cv2.imwrite(f"{save_path}", image)
        print(f"Frame berhasil tersimpan di {save_path}")

    def create_dir(self, path):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print("Berhasil membuat folder", path)
        except OSError:
            print(f"ERROR: creating directory with name {path}")

    # Local Binary Patterns
    def __get_pixel(self, img, center, x, y):
        new_value = 0

        if x < 0 or y < 0 or x >= 256 or y >= 256:
            new_value = 0
        else:
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
        for cat in ["real", "replay", "print"]:
            self.create_dir(f"{self.save_path}\\{cat}")
            images = glob(f"{self.source_path}\\{cat}\\*")

            for idx, frame in enumerate(images):
                image = cv2.imread(frame)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_lbp = np.zeros((self.img_size, self.img_size), np.uint8)

                for l in range(0, self.img_size):
                    for m in range(0, self.img_size):
                        img_lbp[l, m] = self.__lbp_calculated_pixel(image, l, m)

                self.__save_frame(img_lbp, self.save_path+"\\"+cat+"\\"+cat+str(idx)+".jpg")

    def calc_glcm_all_agls(self, img, label, props):
        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=8, symmetric=True,
                            normed=True)
        print(glcm, end="=glcm\n\n")

        feature = []
        glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
        for item in glcm_props:
            feature.append(item)
        feature.append(label)

        return feature

    def glcm_preprocessing(self):
        properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        columns, imgs, labels, glcm_all_agls = [], [], [], []
        angles = ['0', '45', '90', '135']

        for name in properties:
            for ang in angles:
                columns.append(name + "_" + ang)
        columns.append("label")

        for cat in ["real", "replay", "print"]:
            image_files = glob(f"{self.save_path}\\{cat}\\*")
            for idx, frame in enumerate(image_files):
                image = cv2.imread(frame)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_glcm = np.zeros((self.img_size, self.img_size), np.uint8)

                for i in range(0, self.img_size):
                    for j in range(0, self.img_size):
                        img_glcm[i, j] = int(math.floor(image[i][j] / 32))

                imgs.append(img_glcm)
                labels.append(cat)

                print(img_glcm, end="=img_glcm\n\n")
                print(cat, end="=cat\n\n")

        for img, label in zip(imgs, labels):
            glcm_all_agls.append(self.calc_glcm_all_agls(img, label, props=properties))

        glcm_df = pd.DataFrame(glcm_all_agls, columns=columns)
        glcm_df.to_csv(f"{self.save_path}\\glcm24.csv")
        print(f"CSV berhasil tersimpan di {self.save_path}\\glcm24.csv\n")


if __name__ == '__main__':
    model1 = Model(
        nama="model1",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_1",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_1",
    )
    # model1.lbp_preprocessing()
    model1.glcm_preprocessing()

    model2 = Model(
        nama="model2",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_2",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_2",
    )
    # model2.lbp_preprocessing()
    model2.glcm_preprocessing()

    model3 = Model(
        nama="model3",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_3",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_3",
    )
    # model3.lbp_preprocessing()
    model3.glcm_preprocessing()

    model4 = Model(
        nama="model4",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_4",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_4",
    )
    # model4.lbp_preprocessing()
    model4.glcm_preprocessing()

    model5 = Model(
        nama="model5",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_5",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_5",
    )
    # model5.lbp_preprocessing()
    model5.glcm_preprocessing()

    model6 = Model(
        nama="model6",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_6",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_6",
    )
    # model6.lbp_preprocessing()
    model6.glcm_preprocessing()

    model7 = Model(
        nama="model7",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_7",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_7",
    )
    # model7.lbp_preprocessing()
    model7.glcm_preprocessing()

    model8 = Model(
        nama="model8",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_8",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_8",
    )
    # model8.lbp_preprocessing()
    model8.glcm_preprocessing()

    model9 = Model(
        nama="model9",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_9",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_9",
    )
    # model9.lbp_preprocessing()
    model9.glcm_preprocessing()

    model10 = Model(
        nama="model10",
        source_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_10",
        img_size=256,
        save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_10",
    )
    # model10.lbp_preprocessing()
    model10.glcm_preprocessing()
