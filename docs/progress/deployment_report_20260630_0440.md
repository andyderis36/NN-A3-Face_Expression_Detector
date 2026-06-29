# Laporan Progres Deployment — WebRTC Fallback & Clean Separation
**Tanggal Laporan:** 2026-06-30 04:40 WITA (UTC+8)

---

## 1. Ringkasan Pencapaian

| Item | Status | Detail |
|:-----|:-------|:-------|
| Diagnosa model ONNX corrupted | ✅ Selesai | File cuma 132 bytes (Git LFS pointer) — `git lfs pull` sukses dapetin file asli (2.48 MB ONNX, 7.48 MB H5) |
| WebRTC di cloud | ❌ Gagal total | Semua TURN approach (HF native, Cloudflare, Metered.ca) gagal di HF Spaces |
| Camera-input auto-capture | ✅ Berhasil | `st.camera_input()` + JS auto-click tiap 300ms — works di HF Spaces |
| Pemisahan app.py vs appLocal.py | ✅ Selesai | `app.py` = cloud (camera-input), `appLocal.py` = local (WebRTC) |
| Update .gitignore | ✅ Selesai | Tambah `*.log`, `streamlit_local*` |

---

## 2. Kronologi Masalah ONNX Model

### Gejala
```
Failed to load resources: [ONNXRuntimeError] : 7 : INVALID_PROTOBUF : 
Load model from ...\output\emotion_stress_cnn.onnx failed: Protobuf parsing failed.
```

### Root Cause
File `.onnx` dan `.h5` di `output/` dan `Pack01b_DeploymentCode/` adalah **Git LFS pointer files** — isinya cuma:
```
version https://git-lfs.github.com/spec/v1
oid sha256:175dbfdef5fb171bbf1228107a8b3b454347d99a03b9b2
```
Bukan model asli. `git lfs pull` dibutuhkan untuk mendownload binary aktual.

### Solusi
```bash
git lfs pull
```
| File | Sebelum | Sesudah |
|:-----|:--------|:--------|
| `output/emotion_stress_cnn.onnx` | 132 bytes | 2.48 MB ✅ |
| `output/emotion_stress_cnn.h5` | 132 bytes | 7.48 MB ✅ |
| `Pack01b_DeploymentCode/stress_detector_cnn.h5` | 133 bytes | 10.1 MB ✅ |

---

## 3. Kronologi Masalah WebRTC (Final Verdict)

### Latar Belakang
Sejak awal, WebRTC (`streamlit-webrtc`) gagal di HF Spaces & Streamlit Cloud karena blokade UDP outbound. Beberapa approach telah dicoba:

| Approach | Status | Detail |
|:---------|:-------|:-------|
| **Attempt 0:** Fallback ke upload/snapshot only | ✅ Work | Tapi user kehilangan live webcam experience |
| **Attempt 1:** `fastrtc` library (get_hf_turn_credentials) | ❌ Gagal | Konflik `wasm_utils` dengan gradio bawaan HF |
| **Attempt 2:** Metered.ca TURN TCP relay | ❌ Gagal | DNS `openrelay.metered.ca` mungkin diblokir |
| **Attempt 3:** Cloudflare TURN via `turn.fastrtc.org` | ❌ Gagal | DNS `Temporary failure in name resolution` |
| **Attempt 4:** `get_hf_ice_servers()` + `?transport=tcp` | ❌ Gagal | "Connection is taking longer than expected" — TURN TCP tetap fails |

### Final Verdict — WebRTC Tidak Viable di Cloud Serverless
Setelah 5 approach berbeda, **WebRTC via TURN relay dinyatakan tidak viable** di HF Spaces/Streamlit Cloud:
- HF Spaces blok outbound UDP → WebRTC P2P gagal
- DNS internal HF gagal resolve `turn.fastrtc.org` → TURN credential gagal
- Metered.ca TCP TURN juga gagal (mungkin juga DNS block)
- `get_hf_ice_servers()` sukses ambil credential, tapi TURN connection tetap timeout

---

## 4. Solusi Final — Camera-Input Auto-Capture Loop

### Arsitektur
```
Browser                          HF Spaces Server
  │                                    │
  ├─ st.camera_input() preview ────────┤ (widget render)
  │                                    │
  ├─ JS auto-click "Take Photo" ──────┤ (every 300ms)
  │      tiap 300ms                    │
  │                                    │
  │    [frame captured] ──────────────>│ process_cloud_frame()
  │                                    │   ├─ cv2 decode BGR
  │    <── st.image(processed) ────────│   ├─ face_cascade.detectMultiScale
  │                                    │   ├─ ONNX inference (7 emotions)
  │    [telemetry update]              │   ├─ draw bounding box + stress HUD
  │                                    │   └─ return RGB frame
  │    ─── loop ──────────────────────>│ (setInterval + rerun cycle)
```

### Keuntungan
- ✅ **100% works di cloud** — pake HTTP, bukan WebRTC/UDP
- ✅ **Live camera experience** — 3-4 fps, cukup buat deteksi real-time
- ✅ **Processing sama persis** — pake ONNX + Haar Cascade yang sama
- ✅ **Telemetry tetap jalan** — stress counter, alert, dashboard
- ✅ **Zero extra dependency** — cuma `st.camera_input()` standar

### Kekurangan
- ⚠️ Latency lebih tinggi dari WebRTC (~300ms vs ~100ms)
- ⚠️ Frame rate lebih rendah (3-4 fps vs 15-30 fps WebRTC)
- ⚠️ `st.rerun()` tiap capture → overhead rendering page

---

## 5. Separation of Concerns — app.py vs appLocal.py

Sebelumnya `app.py` mencampur cloud detection + TURN config + WebRTC dalam satu file. Sekarang dipisah:

| File | Tujuan | Webcam Method | Cloud Detection |
|:-----|:-------|:--------------|:----------------|
| **`app.py`** | Cloud deployment (HF Spaces) | `st.camera_input()` + JS auto-capture | ✅ Ada + DNS diagnostics |
| **`appLocal.py`** | Local development | `webrtc_streamer()` (STUN P2P) | ❌ Tidak ada (pure lokal) |

### `appLocal.py` — Clean Local Version
- Tidak ada `get_cloud_rtc_config()`, `process_cloud_frame()`, cloud detection
- WebRTC langsung dengan STUN (`stun:stun.l.google.com:19302`)
- Semua fitur lain identik (upload, analytics dashboard)

---

## 6. Status Fitur Aplikasi

| Fitur | Local (`appLocal.py`) | Cloud (`app.py`) | Catatan |
|:------|:---------------------|:-----------------|:--------|
| **🎥 Real-time Webcam** | ✅ WebRTC (STUN P2P) | ✅ Camera-input auto-capture | Method beda, experience serupa |
| **📁 Upload Image/Video** | ✅ Berjalan Lancar | ✅ Berjalan Lancar | Sama di kedua versi |
| **📸 Camera Snapshot** | ✅ Berjalan Lancar | ✅ Berjalan Lancar | Sama di kedua versi |
| **📊 Analytics Dashboard** | ✅ Berjalan Lancar | ✅ Berjalan Lancar | Sama di kedua versi |
| **Stress Detection & Logging** | ✅ Berjalan Lancar | ✅ Berjalan Lancar | Sama di kedua versi |

---

## 7. File yang Berubah

```
app.py
    ├─ Hapus: WebRTC cloud path
    ├─ Tambah: camera-input auto-capture loop (cloud)
    ├─ Tambah: process_cloud_frame() function
    ├─ Tambah: cloud stress state (st.session_state)
    └─ Ubah: telemetry baca dari cloud_stress saat di cloud

appLocal.py (NEW)
    ├─ Copy dari app.py, strip semua cloud code
    ├─ WebRTC-only (langsung STUN)
    └─ Telemetry langsung dari StressState

.gitignore
    └─ Tambah: *.log, streamlit_local*
```

---

## 8. Rencana Selanjutnya

1. **Monitoring** — Pantau HF Spaces app berjalan stabil dengan camera-input loop
2. **Performance tuning** — Optimasi frame rate jika perlu (kurangi interval JS atau percepat processing)
3. **UI improvement** — Tambah indikator fps/status di cloud webcam view
4. **Revoke hardcoded token** — `docs/credentials.md` masih mengandung HF write token (pindah ke secrets)
