import os

import numpy
from flask import render_template, url_for, flash, redirect, request, g, Response
from sistempresensi import app, db, global_variables as gvar
import datetime
import cv2
import time
import face_recognition
import jyserver.Flask as jsf
import glob
import numpy as np
from sqlalchemy import or_, select, and_, update, func
from sistempresensi.forms import MulaiKelasForm, LihatLaporanForm
from sistempresensi.models import Dosen, Mahasiswa, Matkul, Kelas, Enrollment, Kehadiran


@jsf.use(app)
class App:
    def __init__(self):
        self.logger = ""

    @jsf.task
    def main(self):
        try:
            self.js.dom.logger.innerHTML = gvar['logger']
            time.sleep(0.2)
        except TimeoutError:
            print("timeout error 2")


def refresh_kelas():
    try:
        batas = datetime.datetime.now() + datetime.timedelta(hours=-5)
        kelas = Kelas.query.filter_by(selesai=None).all()
        for i in kelas:
            if i.mulai < batas:
                i.selesai = i.mulai + datetime.timedelta(hours=5)
        db.session.commit()
    except Exception as e:
        print("Error pada refresh_kelas: ", e)


@app.errorhandler(404)
def not_found(e='Url not found!'):
    return render_template('error.html', e=e, title="Error"), 404


@app.route("/")
@app.route("/home")
def home():
    try:
        refresh_kelas()
        ongoing_kelas = Kelas.query.filter(Kelas.selesai == None).all()
        recent_kelas = Kelas.query.filter(datetime.datetime.now() + datetime.timedelta(minutes=-30) < Kelas.selesai). \
            order_by(Kelas.selesai.desc()).all()
        today_kelas = Matkul.query.filter(Matkul.hari == datetime.datetime.now().strftime("%w")).all()
        return render_template('home.html', today_kelas=today_kelas, ongoing_kelas=ongoing_kelas,
                               recent_kelas=recent_kelas, title="Home")
    except Exception as e:
        print("Error pada home()")


@app.route("/mulai-kelas", methods=["GET", "POST"])
def mulaikelas():
    try:
        form = MulaiKelasForm()
        if form.validate_on_submit():
            db.session.execute(select(Matkul).filter_by(kode_matkul_group=form.kode_matkul_group.data)).scalar_one()\
                .total_pertemuan += 1
            nomor_pertemuan = str(
                Matkul.query.filter(Matkul.kode_matkul_group == form.kode_matkul_group.data).first().total_pertemuan)
            kode_kelas = form.kode_matkul_group.data + nomor_pertemuan
            db.session.add(Kelas(kode_kelas=kode_kelas, matkul_group=form.kode_matkul_group.data, keterangan="",
                                 mulai=datetime.datetime.now()))
            db.session.commit()
            os.mkdir("sistempresensi/static/kehadiran/" + kode_kelas)
            return redirect(url_for("kelas", id=kode_kelas))

        datas = Matkul.query.filter_by(tahun_ajaran=gvar['tahun_ajaran'], semester=gvar['semester']).all()
        return render_template('mulai-kelas.html', datas=datas, form=form, title="Mulai Kelas")
    except Exception as e:
        print("Error pada mulai-kelas")


# @app.route("/verifikasi/<id>", methods=["GET", "POST"])
@app.route("/kelas/<id>", methods=["GET", "POST"])
def kelas(id):
    refresh_kelas()
    kelas = Kelas.query.filter_by(kode_kelas=id).first_or_404("Kelas tidak ditemukan")
    enrollment = Enrollment.query.filter_by(matkul_group=kelas.matkul_group).order_by(Enrollment.nim).all()
    kehadiran = Kehadiran.query.filter(Kehadiran.kelas == id).all()

    return App.render(
        render_template("kelas.html", kehadiran=kehadiran, kelas=kelas, enrollment=enrollment, title="Kelas"))


@app.route('/video/<kode_kelas>')
def video(kode_kelas):
    try:
        return Response(generate_frames(kode_kelas), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print("Error pada video()")


def generate_frames(kode_kelas):
    try:
        with app.app_context():
            known_face_encodings, known_face_names = training(kode_kelas)
            video_capture = cv2.VideoCapture(0)
            history = [0] * len(known_face_encodings)

            while True:
                # Grab a single frame of video
                ret, frame = video_capture.read()
                frame_toshow = frame

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    gvar['logger'] = "Tidak dapat membuka kamera. Pastikan kamera aktif dan tidak sedang " \
                                     "digunakan aplikasi lain."
                    App.main()
                    video_capture.release()
                    video_capture.read()
                    break

                if known_face_encodings: # Jika ada yg enroll
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]

                        face_names.append(name)
                        print(face_names)

                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4

                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                        frame_toshow = cv2.rectangle(frame, (0, 0), (1000, 45), (216, 117, 2), cv2.FILLED)
                        if name != 'Unknown':
                            cek_kehadiran = Kehadiran.query.filter_by(kelas=kode_kelas, nim=name).all()
                            if not cek_kehadiran:  # Jika belum hadir
                                gvar['logger'] = name + " hadir"
                                App.main()

                                db.session.add(Kehadiran(waktu=datetime.datetime.now(), kelas=kode_kelas, nim=name,
                                                         foto_path=name+".jpg"))
                                db.session.commit()

                                cv2.imwrite("sistempresensi/static/kehadiran/"+kode_kelas+"/"+name + ".jpg", frame)
                            else:
                                cv2.putText(frame_toshow, name + " telah tercatat", (5, 30), cv2.FONT_HERSHEY_PLAIN,
                                            2.0, (247, 247, 247), 2)
                        else:
                            cv2.putText(frame_toshow, "Wajah tidak dikenali", (5, 30), cv2.FONT_HERSHEY_PLAIN, 2.0,
                                        (247, 247, 247), 2)

                if cv2.waitKey(1) == 32:
                    video_capture.release()
                    break

                ret, buffer = cv2.imencode('.jpg', frame_toshow)
                frame_toshow = buffer.tobytes()

                yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_toshow + b'\r\n'
    except Exception as e:
        print("Error pada generate frames(haha): ", e)


def training(kode_kelas):
    try:
        kelas = Kelas.query.filter_by(kode_kelas=kode_kelas).first_or_404()
        enrollment = Enrollment.query.filter_by(matkul_group=kelas.matkul_group).order_by(Enrollment.nim).all()
        foto_path = []
        nama = []

        for i in enrollment:
            foto_path.append("sistempresensi/static/mahasiswa/" + i.data_mahasiswa.foto_path)
            nama.append(i.data_mahasiswa.nama)

        # dosen_path = glob.glob("sistempresensi/static/dosen/*")
        # mahasiswa_path = glob.glob("sistempresensi/static/mahasiswa/*")
        known_face_encodings = []
        known_face_names = []

        # for i in np.concatenate((dosen_path, mahasiswa_path)):
        for i in range(len(foto_path)):
            file_name = foto_path[i].split("/")[3].split(".")[0]
            image = face_recognition.load_image_file(foto_path[i])
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            known_face_names.append(file_name)

        print('Learned encoding for', len(known_face_encodings), 'images.')

        gvar['logger'] = "Selesai melatih " + str(len(known_face_encodings)) + " wajah"
        App.main()

        return known_face_encodings, known_face_names
    except Exception as e:
        print("Error pada training()")


@app.route("/kelas/<id>/ended", methods=['POST'])
def end_kelas(id):
    try:
        kelas = Kelas.query.filter_by(kode_kelas=id).first()
        if kelas != None:
            if kelas.selesai == None:
                kelas.selesai = datetime.datetime.now()
        db.session.commit()
        return redirect(url_for('home'))
    except Exception as e:
        print("Error pada end_kelas")


def refresh_kehadiran(a):
    kehadiran = Kehadiran.query.filter(Kehadiran.kelas == a).all()
    return kehadiran


@app.route("/about")
def about():
    return render_template('about.html', title='About')


@app.route("/dosen")
def dosen():
    datas = Dosen.query.all()
    return render_template('dosen.html', datas=datas, title="Dosen")


@app.route("/dosen/<id>")
def detaildosen(id):
    data = Dosen.query.filter_by(nip=id).first_or_404("Dosen tidak ditemukan")
    matkul = Matkul.query.filter(Matkul.dosen == id, Matkul.tahun_ajaran == gvar['tahun_ajaran'],
                                 Matkul.semester == gvar["semester"]).all()
    return render_template('detail-dosen.html', data=data, matkul=matkul, namahari=gvar['hari'], title="Detail Dosen")


@app.route("/mahasiswa")
def mahasiswa():
    datas = Mahasiswa.query.all()
    return render_template('mahasiswa.html', datas=datas, title="Mahasiswa")


@app.route("/mahasiswa/<id>")
def detailmahasiswa(id):
    data = Mahasiswa.query.filter_by(nim=id).first_or_404("Dosen tidak ditemukan")
    enrollment = Enrollment.query.join(Matkul).filter(Enrollment.nim == id, Matkul.tahun_ajaran == gvar['tahun_ajaran'],
                                                      Matkul.semester == gvar["semester"]).all()
    return render_template('detail-mahasiswa.html', data=data, enrollment=enrollment, namahari=gvar['hari'],
                           title="Detail Mahasiswa")


@app.route("/matakuliah")
def matakuliah():
    datas = Matkul.query.filter(Matkul.tahun_ajaran == gvar['tahun_ajaran'], Matkul.semester == gvar["semester"]).all()
    return render_template('matakuliah.html', datas=datas, namahari=gvar['hari'], Dosen=Dosen, title="Mata Kuliah")


@app.route("/matakuliah/<id>")
def detailmatakuliah(id):
    data = Matkul.query.filter_by(kode_matkul_group=id).first()
    enrollment = Enrollment.query.filter_by(matkul=id, tahun_ajaran=gvar['tahun_ajaran'],
                                            semester=gvar["semester"]).all()
    return render_template('detail-matakuliah.html', data=data, namahari=gvar['hari'], enrollment=enrollment,
                           title="Mata Kuliah")


@app.route("/laporan-kehadiran")
def laporankehadiran1():
    datas = Matkul.query.group_by(Matkul.tahun_ajaran, Matkul.semester).order_by(Matkul.id.desc()).all()
    return render_template('laporankehadiran1.html', datas=datas, title="Laporan Kehadiran")


@app.route("/laporan-kehadiran/<tahun_ajaran>/<semester>")
def laporankehadiran2(tahun_ajaran, semester):
    datas = Matkul.query.group_by(Matkul.kode_matkul).having(Matkul.tahun_ajaran == tahun_ajaran, Matkul.semester
                                                             == semester).order_by(Matkul.kode_matkul).all()
    if not datas:
        return render_template('error.html', e="Tahun ajaran/Semester tidak valid!", title="Error"), 404
    return render_template('laporankehadiran2.html', datas=datas, title="Laporan Kehadiran")


@app.route("/laporan-kehadiran/<tahun_ajaran>/<semester>/<kode_matkul>")
def laporankehadiran3(tahun_ajaran, semester, kode_matkul):
    datas = Matkul.query.filter(Matkul.tahun_ajaran == tahun_ajaran, Matkul.semester == semester,
                                Matkul.kode_matkul == kode_matkul).order_by(Matkul.group).all()
    if not datas:
        return render_template('error.html', e="Kode matkul tidak valid!", title="Error"), 404
    return render_template('laporankehadiran3.html', datas=datas, title="Laporan Kehadiran")


@app.route("/laporan-kehadiran/<tahun_ajaran>/<semester>/<kode_matkul>/<kelas>")
def laporankehadiran(tahun_ajaran, semester, kode_matkul, kelas):
    kode_matkul_group = tahun_ajaran + semester + kode_matkul + kelas

    datas = Matkul.query.filter(Matkul.kode_matkul_group == kode_matkul_group).all()
    if not datas:
        return render_template('error.html', e="Kelas tidak valid!", title="Error"), 404

    kelas_data = Kelas.query.filter(Kelas.matkul_group == kode_matkul_group).all()
    enrollment_data = Enrollment.query.filter(Enrollment.matkul_group == kode_matkul_group).order_by(Enrollment.
                                                                                                     nim).all()
    kehadiran_data = Kehadiran.query.join(Kelas).filter(Kelas.matkul_group == kode_matkul_group).all()

    value = np.empty((), dtype=object)
    value[()] = (0, 0, 0, 0, 0)
    flags = np.full((len(enrollment_data), len(kelas_data)), value, dtype=object)
    total_kehadiran = np.zeros(shape=len(enrollment_data), dtype=numpy.int8)

    for index_i, i in enumerate(kelas_data):
        for index_j, j in enumerate(enrollment_data):
            for k in kehadiran_data:
                if k.nim == j.nim and k.kelas == i.kode_kelas:
                    total_kehadiran[index_j - 1] += 1
                    delta_time = k.waktu - i.mulai
                    flags[index_j - 1][index_i - 1] = (i.kode_kelas, 1, k.waktu, int(delta_time.total_seconds()),
                                                       k.foto_path)

    for i in range(len(total_kehadiran)):
        total_kehadiran[i] = int(total_kehadiran[i] / len(kelas_data) * 100)

    return render_template('laporan-kehadiran.html', total_kehadiran=total_kehadiran, data_enrollment=enrollment_data,
                           flags=flags, title="Laporan Kehadiran")
