from sistempresensi import app, db, global_variables as gvar
from sistempresensi.models import Dosen, Mahasiswa, Matkul, Enrollment, Kelas, Kehadiran

if __name__ == "__main__":

    with app.app_context():
        Mhs = Mahasiswa.query.all()
        print(Mhs)

        Mhs = Dosen.query.all()
        print(Mhs)

        Mhs = Matkul.query.all()
        print(Mhs)

        Mhs = Enrollment.query.all()
        print(Mhs)

        Mhs = Kelas.query.all()
        print(Mhs)

        Mhs = Kehadiran.query.all()
        print(Mhs)

        print(gvar)
        # users = Matakuliah.query.filter_by(sks=3).all()
        # print(users)


