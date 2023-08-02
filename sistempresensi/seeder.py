import datetime
import pandas as pd

from sistempresensi import app, db, global_variables as gvar
from sistempresensi.models import Dosen, Mahasiswa, Matkul, Enrollment, Kehadiran, Kelas

if __name__ == "__main__":
    with app.app_context():
        # Drop all table
        db.drop_all()

        # Create all table
        db.create_all()

        file_path = 'static/seeders/seeders.xlsx'

        # Insert Data Mahasiswa
        df = pd.read_excel(file_path, sheet_name='mahasiswa')
        for index, row in df.iterrows():
            db.session.add(Mahasiswa(nim=row.nim, nama=row.nama, email=row.email, telp=row.telp, foto_path=row.foto_path))

        # Insert Data Dosen
        df = pd.read_excel(file_path, sheet_name='dosen')
        for index, row in df.iterrows():
            db.session.add(Dosen(nip=row.nip, nama=row.nama, email=row.email, telp=row.telp, foto_path=row.foto_path))

        # Insert Data Matkul_Kelas
        df = pd.read_excel(file_path, sheet_name='matkul')
        for index, row in df.iterrows():
            hh = int(row.jam.split(":")[0])
            mm = int(row.jam.split(":")[1])
            db.session.add(Matkul(kode_matkul_group=row.kode_matkul_group, kode_matkul=row.kode_matkul, nama_matkul=row.nama_matkul, sks=row.sks, tahun_ajaran=row.tahun_ajaran, semester=row.semester,
                                      group=row.group, dosen=row.dosen, hari=row.hari, jam=datetime.time(hh, mm), ruang=row.ruang,
                                      total_pertemuan=row.total_pertemuan))

        # Insert Data Enrollment
        df = pd.read_excel(file_path, sheet_name='enrollment')
        for index, row in df.iterrows():
            db.session.add(Enrollment(nim=row.nim, matkul_group=row.matkul_group))
        #
        #
        # # # Insert Data Kehadiran
        # # db.session.add(Kehadiran(nim=1908561047, kode_kelas="202220232IF22613011A8", waktu=datetime.datetime.now()))

        db.session.commit()
