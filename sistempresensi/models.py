from sistempresensi import db
from sqlalchemy.sql import func


class Mahasiswa(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nim = db.Column(db.BigInteger, unique=True, nullable=False)
    nama = db.Column(db.String, nullable=False)
    email = db.Column(db.String)
    telp = db.Column(db.BigInteger)
    foto_path = db.Column(db.String, default="default.jpg")
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())

    def __repr__(self):
        return f"\nMahasiswa(id: {self.id}, nim: {self.nim}, nama: {self.nama}', 'email: {self.email}', " \
               f"'telp: {self.telp}', foto_path: '{self.foto_path}', created_at: '{self.created_at}')"


class Dosen(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nip = db.Column(db.BigInteger, unique=True, nullable=False)
    nama = db.Column(db.String, nullable=False)
    email = db.Column(db.String)
    telp = db.Column(db.BigInteger)
    foto_path = db.Column(db.String, default="nophotoprofile.jpeg")
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())

    def __repr__(self):
        return f"\nDosen(id: {self.id}, nip: {self.nip}, nama: {self.nama}, email: {self.email}, " \
                    f"telp: {self.telp}, foto_path: {self.foto_path}, created_at: {self.created_at})"


class Matkul(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kode_matkul_group = db.Column(db.String(), unique=True, nullable=False)
    tahun_ajaran = db.Column(db.Integer, nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    kode_matkul = db.Column(db.String(), nullable=False)
    group = db.Column(db.String(), nullable=False, default='A')
    nama_matkul = db.Column(db.String(), nullable=False)
    sks = db.Column(db.Integer, nullable=False)
    dosen = db.Column(db.BigInteger, db.ForeignKey('dosen.nip'), nullable=False)
    hari = db.Column(db.Integer)    # number of weekdays
    jam = db.Column(db.Time())
    ruang = db.Column(db.String)
    total_pertemuan = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, server_default=func.now())
    data_dosen = db.relationship('Dosen', lazy=True)

    def __repr__(self):
        return f"\nMatkul(id: {self.id}, kode_matkul_group: {self.kode_matkul_group}, " \
               f"tahun_ajaran: {self.tahun_ajaran}, semester: {self.semester}, kode_matkul: {self.kode_matkul}, " \
               f"group: {self.group}, nama_matkul: {self.nama_matkul}, sks: {self.sks}, " \
               f"dosen: {self.dosen}, hari: {self.hari}, jam: {self.jam}, ruang: {self.ruang}, " \
               f"total_pertemuan: {self.total_pertemuan}, created_at: {self.created_at}, " \
               f"data_dosen: {self.data_dosen})"


class Enrollment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nim = db.Column(db.BigInteger, db.ForeignKey('mahasiswa.nim'), nullable=False)
    matkul_group = db.Column(db.String, db.ForeignKey('matkul.kode_matkul_group'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    data_mahasiswa = db.relationship('Mahasiswa', lazy=True)
    data_matkul = db.relationship('Matkul', lazy=True)

    def __repr__(self):
        return f"\nEnrollment(id: {self.id}, nim: {self.nim}, matkul_group: {self.matkul_group}, " \
               f"created_at: {self.created_at}, data_mahasiswa: {self.data_mahasiswa}, " \
               f"data_matkul: {self.data_matkul})"


class Kelas(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kode_kelas = db.Column(db.String, unique=True, nullable=False)
    matkul_group = db.Column(db.String, db.ForeignKey('matkul.kode_matkul_group'), nullable=False)
    mulai = db.Column(db.DateTime, nullable=False, server_default=func.now())
    selesai = db.Column(db.DateTime)
    keterangan = db.Column(db.Text)
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    data_matkul = db.relationship('Matkul', lazy=True)

    def __repr__(self):
        return f"\nKelas(id: {self.id}, kode_kelas: {self.kode_kelas}, matkul_group: {self.matkul_group}," \
               f" mulai: {self.mulai}, selesai: {self.selesai}, keterangan: {self.keterangan}, " \
               f"created_at: {self.created_at}, data_matkul: {self.data_matkul})"


class Kehadiran(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    kelas = db.Column(db.Integer, db.ForeignKey('kelas.kode_kelas'), nullable=False)
    nim = db.Column(db.BigInteger, db.ForeignKey('mahasiswa.nim'), nullable=False)
    waktu = db.Column(db.DateTime, nullable=False, server_default=func.now())
    foto_path = db.Column(db.String, default="absen.jpeg")
    created_at = db.Column(db.DateTime, nullable=False, server_default=func.now())
    data_mahasiswa = db.relationship('Mahasiswa', lazy=True)
    data_kelas = db.relationship('Kelas', lazy=True)

    def __repr__(self):
        return f"\nKehadiran(id: { self.id}, kelas: {self.kelas}, nim: {self.nim}, waktu: {self.waktu}, " \
               f"foto_path: {self.foto_path}, created_at: {self.created_at}, data_mahasiswa: {self.data_mahasiswa}," \
               f" data_kelas: {self.data_kelas})"
