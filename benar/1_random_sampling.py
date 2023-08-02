import cv2
import os
import face_recognition
import random
import datetime
from glob import glob


def create_dir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print("Berhasil membuat folder", path)
    except OSError:
        print(f"ERROR: creating directory with name {path}")


def save_frame(video_path, save_dir):
    name = video_path.split(".")[0].split("\\")[-1]
    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        idx+=1
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

    random_number = random.randint(1, idx)
    print(random_number)

    cap = cv2.VideoCapture(video_path)
    idx=0
    while True:
        idx+=1
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        if idx == random_number:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                for top, right, bottom, left in face_locations:
                    crop_img = frame[top:bottom, left:right]
                    crop_img = cv2.resize(crop_img, (256, 256))
                    cv2.imwrite(f"{save_dir}\\{name}_{idx}.jpg", crop_img)
                    print(f"Frame berhasil tersimpan di {save_dir}\\{name}_{idx}")
                    return 0
            else:
                print("tidak ada wajah")
                random_number+=1


if __name__ == '__main__':
    print("Mulai: ", datetime.datetime.now())
    for i in range(1, 11):
        for cat in ["real", "replay", "print"]:
            video_paths = glob(f"C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\0-video\\{cat}\\*")
            save_dir = f"C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\random_sample_{i}\\{cat}"
            create_dir(save_dir)

            for vid in video_paths:
                save_frame(vid, save_dir)

    print("Selesai: ", datetime.datetime.now())
