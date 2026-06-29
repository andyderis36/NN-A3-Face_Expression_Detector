# Facial Images-based Stress & Emotion Detection (STINK3014 A252 - Assignment 3)

Dokumentasi lengkap untuk repositori tugas "FACIAL IMAGES-BASED STRESS AND EMOTION DETECTION USING DEEP LEARNING TECHNIQUE".

## Ringkasan proyek
Sistem real-time untuk mendeteksi emosi dan status stres dari citra wajah menggunakan Convolutional Neural Network (CNN). Sistem ini mencakup komponen:
- Pelatihan model (baseline) dari FER2013
- Deployment real-time menggunakan webcam dan Haar cascade
- Ekstensi konseptual untuk aplikasi dunia nyata (Part II tugas)

Referensi instruksi tugas: `STINK3014_A252_Assignment03.pdf` (lihat file di root repo).

## Struktur repositori
- `STINK3014_A252_Assignment03.pdf` - Instruksi tugas lengkap.
- `Pack01a_BaselineCode/` - Kode & data untuk melatih model (mengandung `fer2013.csv` dan `STINK3014_CaseStudy_FacialExpressionDetection_Dev.py`).
- `Pack01b_DeploymentCode/` - Kode untuk deployment & demo real-time (`STINK3014_CaseStudy_FacialExpressionDetection_App.py`, dan file bantuan seperti `haarcascade_frontalface_default.xml` atau model `stress_detector_cnn.h5`).
- `Pack02_SUEQ tool/` - Alat S-UEQ (Short UEQ) untuk evaluasi pengalaman pengguna (PDF & Excel).
- `.venv/` - Virtual environment Python (sudah ada). Gunakan selalu `.venv` untuk menjalankan skrip dan instal paket.
- `.gitignore` - file gitignore

## Tujuan dokumentasi ini
- Menyediakan panduan cepat untuk men-setup environment dan menjalankan skrip di Windows
- Menjelaskan file penting dan alur kerja: pra-pemrosesan, pelatihan, penyimpanan model, deployment, logging, dan evaluasi
- Menyertakan catatan etika/privasi dan persyaratan laporan/presentasi

## Persyaratan & Dependensi (disarankan)
Disarankan membuat dan menggunakan virtual environment `.venv` yang sudah ada.
Saran paket Python (tambahkan ke `requirements.txt` jika diinginkan):
- tensorflow (sesuaikan versi yang tersedia di sistem)
- opencv-python
- numpy
- pandas
- matplotlib
- scikit-learn
- pypdf
- playsound (opsional, untuk audio) atau gunakan `winsound` pada Windows

Contoh entry `requirements.txt` (opsional):
```
tensorflow
opencv-python
numpy
pandas
matplotlib
scikit-learn
pypdf
playsound
```

## Penggunaan (.venv) — Windows
Semua perintah berikut diasumsikan dijalankan di root repo (lokasi file `README.md`). Gunakan terminal PowerShell atau cmd.exe.

1) Mengaktifkan virtual environment (PowerShell):
```
.\.venv\Scripts\Activate.ps1
```
Atau (cmd.exe):
```
.\.venv\Scripts\activate.bat
```

2) Menginstal dependensi (jika perlu):
```
.\.venv\Scripts\pip.exe install -r requirements.txt
```
Atau menginstal paket satu per satu:
```
.\.venv\Scripts\pip.exe install tensorflow opencv-python numpy pandas matplotlib scikit-learn pypdf playsound
```

3) Menjalankan skrip pelatihan (baseline):
```
.\.venv\Scripts\python.exe "Pack01a_BaselineCode\STINK3014_CaseStudy_FacialExpressionDetection_Dev.py"
```
Skrip tersebut memproses `fer2013.csv`, membuat dataset stress vs calm, melatih CNN sederhana, dan menyimpan model sebagai `stress_detector_cnn.h5`.

4) Menjalankan aplikasi deployment (webcam):
```
.\.venv\Scripts\python.exe "Pack01b_DeploymentCode\STINK3014_CaseStudy_FacialExpressionDetection_App.py"
```
Pastikan file `stress_detector_cnn.h5` berada di folder yang sama (atau sesuaikan path di skrip). Tekan `q` untuk keluar.

## Deskripsi file penting
- `Pack01a_BaselineCode/STINK3014_CaseStudy_FacialExpressionDetection_Dev.py`
  - Baca `fer2013.csv`, convert string pixel ke array 48x48 grayscale, buat label stress vs calm
  - Membangun model CNN (contoh: 2 conv layer -> pooling -> dense -> softmax), melatih dan menyimpan model
  - Catatan: skrip saat ini memakai 1 epoch (untuk demonstrasi). Untuk eksperimen, tingkatkan jumlah epoch dan pertimbangkan augmentasi/hyperparameter tuning

- `Pack01b_DeploymentCode/STINK3014_CaseStudy_FacialExpressionDetection_App.py`
  - Memuat `stress_detector_cnn.h5`, menangkap frame webcam, mendeteksi wajah dengan Haar cascade, mengekstrak ROI, melakukan prediksi, menampilkan hasil pada frame
  - Perlu penambahan fitur: audio beep alert saat deteksi stres tinggi, dan logging kejadian stres ke CSV

- `Pack02_SUEQ tool/`
  - S-UEQ tools untuk ujicoba UX (kuesioner + analisis). Gunakan untuk bagian Part II tugas.

## Rekomendasi perubahan & TODO penting
1. Tambahkan `requirements.txt` yang sesuai dengan versi TensorFlow yang digunakan.
2. Perbaikan `App.py`:
   - Tambah logging ke CSV (timestamp, predicted_label, confidence, frame_count atau counter)
   - Tambah audio alert: gunakan modul `winsound.Beep(frequency, duration)` di Windows atau `playsound` untuk file .wav
   - Tambah logic untuk mencegah spam alert (mis. deteksi stres selama N frame berturut-turut sebelum memicu alarm)
3. Tambah evaluasi: simpan plot akurasi & loss (matplotlib) ke folder `results/` dan ekspor confusion matrix (CSV / gambar)
4. Buat notebook atau skrip untuk analisis S-UEQ dari `Pack02_SUEQ tool` jika perlu.

Contoh tambahan sederhana (logging + beep) untuk `App.py` (pseudo):
```py
import csv, time
import winsound

# saat stress terdeteksi berulang
with open('stress_log.csv','a',newline='') as f:
    writer = csv.writer(f)
    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), 'Stressed', confidence])

# beep
winsound.Beep(1000, 500)  # 1000 Hz selama 500 ms
```

## Petunjuk eksperimen & evaluasi
- Pelatihan: gunakan `train_test_split` dan `validation_split`; simpan riwayat training untuk membuat plot akurasi & loss
- Evaluasi stres vs non-stres: ekstrak confusion matrix pada set test (stressed vs calm)
- Metrik yang diminta pada tugas: accuracy, stress detection rate, false alarm rate

## Bagian laporan (sesuai instruksi PDF)
Laporan harus mencakup:
- Penjelasan arsitektur CNN dan alasan pemilihan layer / hyperparameter
- Langkah preprocessing FER2013
- Cara perhitungan "stress level" (mis. jumlah frame berturut-turut dengan label stress)
- Grafik akurasi & loss, confusion matrix, analisis trade-off deteksi vs false alarms
- Part II: ide aplikasi, arsitektur, isu privasi & etika, skenario deployment (edge vs server)
- Hasil S-UEQ (analisis pragmatic & hedonic scores)

## Catatan Etika & Privasi
- Pastikan mendapat informed consent saat mengumpulkan data pengguna nyata
- Agar anonimitas, jangan simpan gambar/raw frame kecuali diperlukan dan terenkripsi
- Terapkan prinsip privacy-by-design; laporkan bagaimana data disimpan, siapa yang punya akses, dan berapa lama disimpan

## Troubleshooting
- Jika webcam tidak terdeteksi: coba ganti index `cv2.VideoCapture(0)` ke `1` atau periksa driver
- Jika model gagal dimuat: pastikan `stress_detector_cnn.h5` ada di path yang sama dengan skrip atau ubah `load_model()` path
- Jika perpustakaan TensorFlow tidak terinstal di `.venv`: aktifkan `.venv` lalu jalankan pip install via `.venv\Scripts\pip.exe`

## Contributors & Kontak
- Student: (sesuaikan dengan nama Anda)
- Instructor: Associate Prof. Dr Azizi Ab Aziz

## Lisensi
Gunakan sesuai kebijakan universitas dan tidak menyalin karya orang lain (plagiarisme dilarang sesuai PDF tugas).

---
Catatan: README ini adalah dokumentasi awal yang lengkap untuk mereproduksi dan mengembangkan tugas. Jika ingin, dokumen dapat diperluas menjadi panduan langkah-demi-langkah untuk eksperimen tertentu (hyperparameter tuning, augmentation, deployment on edge, dsb.).
