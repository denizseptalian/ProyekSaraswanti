# Deteksi Objek YOLOv8 TFLite — Streamlit Cloud

Aplikasi Streamlit untuk deteksi objek real-time menggunakan model YOLOv8 TFLite
(`best_model.tflite`, input 640×640, 6 kelas).

## Fitur
- **Live Webcam** — stream kamera browser via WebRTC, deteksi real-time dengan frame-skipping agar lancar di CPU
- **Upload Video** — proses video per-frame, preview langsung, dan unduh video hasil anotasi
- **Upload Gambar** — deteksi tunggal dengan tabel hasil (kelas, confidence, bounding box)
- Slider confidence & IoU threshold di sidebar

## Struktur
```
├── app.py
├── best_model.tflite
├── requirements.txt      # dependensi Python
├── packages.txt          # dependensi sistem (apt) untuk Streamlit Cloud
└── README.md
```

## Deploy ke Streamlit Cloud
1. Buat repo GitHub baru, push semua file di folder ini (termasuk `best_model.tflite`, ukuran 12 MB — aman, batas GitHub 100 MB).
2. Buka https://share.streamlit.io → **New app**.
3. Pilih repo, branch `main`, main file `app.py`.
4. Klik **Deploy**. Selesai — packages.txt otomatis menginstal libgl1/ffmpeg yang dibutuhkan OpenCV & WebRTC.

## Catatan penting
- **Ganti `CLASS_NAMES` di `app.py`** sesuai urutan kelas pada `data.yaml` dataset Roboflow Anda (tbm-3-merge). Saat ini masih placeholder `kelas_0..kelas_5`.
- Streamlit Cloud gratis berjalan di CPU (1 core). Inferensi ±200–500 ms/frame; slider "Proses setiap N frame" membantu menjaga stream tetap lancar.
- Jika webcam WebRTC tidak tersambung di jaringan tertentu (NAT ketat), tambahkan TURN server pada `rtc_configuration`.

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```
