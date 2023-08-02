from sistempresensi import app
from flask_wtf import FlaskForm
from sistempresensi.models import Matkul
from sistempresensi import global_variables as gvar
from wtforms import StringField, SubmitField, BooleanField, SelectField, SelectMultipleField
from wtforms.validators import DataRequired, Length, Email, EqualTo

with app.app_context():
    matkul = Matkul.query.filter_by(tahun_ajaran=gvar['tahun_ajaran'], semester=gvar['semester']).order_by(Matkul.kode_matkul_group).all()


class MulaiKelasForm(FlaskForm):
    option = [('', "---")]
    for i in matkul:
        option.append((i.kode_matkul_group, f"{i.kode_matkul} - {i.nama_matkul} ({i.group})"))
    kode_matkul_group = SelectField(default="", validators=[DataRequired()], choices=option)
    submit = SubmitField('Submit')


class LihatLaporanForm(FlaskForm):
    option = [('', "Pilih Mata Kuliah")]
    for i in matkul:
        option.append((i.kode_matkul, i.kode_matkul+" - " + i.nama_matkul))
    kode_matkul = SelectField('Masukkan Mata Kuliah', default="", validators=[DataRequired()], choices=option)
    submit = SubmitField('submit')
