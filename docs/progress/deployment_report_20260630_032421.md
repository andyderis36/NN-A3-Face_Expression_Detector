# Laporan Progres Deployment: Real-time Stress Detection
**Tanggal Laporan:** 2026-06-30 03:24:21 (Local Time)

Laporan ini merangkum masalah deployment yang dihadapi, solusi yang telah dicoba, serta status progres dan kendala saat ini pada aplikasi **Student Stress Detection System**.

---

## 1. Kronologi Masalah Utama (UDP & WebRTC)
Aplikasi menggunakan `streamlit-webrtc` untuk fitur **Real-time Webcam**. Fitur ini berjalan sempurna di lingkungan lokal (*localhost*), namun mengalami masalah saat dideploy ke layanan cloud gratisan (**Streamlit Cloud** & **Hugging Face Spaces**):

* **Gejala:** Tampilan webcam hanya berputar (*loading*) saat tombol start ditekan, lalu server mengalami crash/timeout setelah 10 detik.
* **Penyebab Utama:** Protokol WebRTC membutuhkan jalur **UDP** terbuka untuk transmisi data video real-time secara peer-to-peer (P2P). Infrastruktur jaringan pada Streamlit Cloud dan Hugging Face Spaces secara ketat memblokir koneksi port UDP keluar (*outbound UDP*).
* **Solusi Sementara (Bypass):** 
  * Mendeteksi lingkungan server secara otomatis (`SPACE_ID` di HF dan `/mount/src` di Streamlit Cloud).
  * Menyembunyikan opsi Live Webcam di cloud dan memaksa fallback ke mode **Upload Gambar / Video** dan **Snapshot Kamera** (`st.camera_input`) yang menggunakan protokol HTTP/HTTPS biasa.

---

## 2. Percobaan Mengaktifkan Live Webcam via HF TURN Server
Untuk memecahkan batasan UDP di Hugging Face Spaces, kita mencoba mengintegrasikan server relay **TURN (Traversal Using Relays around NAT)** resmi milik Hugging Face yang dikelola lewat library `fastrtc`.

### Langkah-langkah:
1. Menambahkan `fastrtc` ke dalam berkas `requirements.txt`.
2. Menggunakan fungsi `get_hf_turn_credentials()` di `app.py` untuk mengambil kredensial TURN server secara dinamis apabila aplikasi berjalan di Hugging Face Spaces.
3. Membungkus fungsi dengan blok `try-except` agar jika terjadi kegagalan, aplikasi tetap berjalan dengan fallback koneksi STUN standar.

---

## 3. Status dan Kendala Terkini (Log Analisis)
Berdasarkan log startup terakhir pada Hugging Face Spaces, fitur Live Webcam masih belum berfungsi karena dua kendala teknis berikut:

### A. Konflik Versi Library (`fastrtc` vs `gradio`)
* **Error Log:**
  ```text
  Failed to get HF TURN credentials: cannot import name 'wasm_utils' from 'gradio' (/usr/local/lib/python3.12/site-packages/gradio/__init__.py)
  ```
* **Analisis:** Library `fastrtc` mencoba mengimpor utilitas internal `wasm_utils` dari pustaka `gradio`. Namun, runtime Hugging Face memiliki pustaka `gradio` bawaan sistem dengan versi yang berbeda/tidak kompatibel. Hal ini menyebabkan kegagalan impor dan fungsi pencarian TURN server gagal dieksekusi.

### B. Kegagalan Socket Koneksi (Asyncio)
* **Error Log:**
  ```text
  AttributeError: 'NoneType' object has no attribute 'sendto'
  AttributeError: 'NoneType' object has no attribute 'call_exception_handler'
  ```
* **Analisis:** Karena modul TURN server gagal dimuat, sistem mencoba melakukan fallback menggunakan server STUN publik milik Google. Namun, karena pembatasan port UDP di Hugging Face Spaces tetap akhir, paket data STUN diblokir. Hal ini memicu kegagalan soket internal pada pustaka asinkron `aioice`/`aiortc` ketika mencoba melakukan koneksi ulang (*retry transaction*).

---

## 4. Status Fitur Aplikasi saat Ini
| Fitur | Status di Lokal | Status di Cloud (HF / Streamlit) | Catatan |
| :--- | :--- | :--- | :--- |
| **Real-time Webcam** | ✅ Berjalan Lancar | ❌ Tidak Stabil / Crash | Masalah UDP block & konflik package `fastrtc`. |
| **Upload Image/Video** | ✅ Berjalan Lancar | ✅ Berjalan Lancar | Sangat stabil & direkomendasikan untuk demo cloud. |
| **Camera Snapshot** | ✅ Berjalan Lancar | ✅ Berjalan Lancar | Solusi terbaik pengganti live webcam di cloud. |

---

## 5. Rencana Tindak Lanjut Selanjutnya (Opsional)
1. **Penyelarasan Versi Gradio:** Mengunci versi `gradio` yang kompatibel dengan `fastrtc` di berkas `requirements.txt` (contoh: `gradio>=4.0.0`) agar impor `wasm_utils` tidak error.
2. **Penyediaan TURN Server Alternatif:** Jika TURN server HF tetap tidak stabil, alternatif lainnya adalah menggunakan layanan TURN server pihak ketiga yang andal (seperti Twilio Network Traversal API atau Metered.ca berbayar).
3. **Fokus pada Fitur Fallback:** Meningkatkan antarmuka pengguna pada fitur Upload Video dan Snapshot Kamera sebagai sarana utama demo di cloud.
