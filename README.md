# Klasifikasi Penyakit Mata dengan CNN dan Grad-CAM

Aplikasi web untuk mengklasifikasikan citra mata ke dalam empat kondisi: Uveitis, Katarak, Pterygium, dan Normal. Pengguna bisa mengambil foto lewat webcam atau mengunggah gambar. Hasilnya berupa kelas prediksi, tingkat keyakinan model, dan visualisasi Grad-CAM yang menunjukkan bagian gambar yang paling memengaruhi prediksi.

Demo: https://deteksimata-cpcv8jvexq5endpctw8cyy.streamlit.app/
(Di-hosting di Streamlit Community Cloud. Kalau aplikasi sedang tidur, tunggu beberapa saat sampai aktif lagi.)

Proyek ini dikerjakan sebagai skripsi di Program Studi Informatika, Universitas Gunadarma.

## Fitur

- Input gambar lewat webcam atau unggah file (JPG/PNG), bisa dibuka dari laptop maupun HP
- Klasifikasi empat kelas: Uveitis, Katarak, Normal, Pterygium
- Menampilkan confidence dan probabilitas untuk setiap kelas
- Visualisasi Grad-CAM, supaya terlihat area mata mana yang dipakai model untuk memutuskan
- Ada disclaimer bahwa hasilnya bukan diagnosis medis

## Hasil

Model diuji pada 657 gambar yang tidak dipakai saat pelatihan.

| Metrik | Nilai |
| Akurasi data uji | 77% |
| Akurasi validasi | 70,84% |
| Macro F1-score | 0,76 |

Kelas | Precision | Recall | F1-score | Jumlah data uji |
| Uveitis | 0,68 | 0,86 | 0,76 | 189 |
| Katarak | 0,92 | 0,63 | 0,75 | 89 |
| Normal | 0,75 | 0,81 | 0,78 | 188 |
| Pterygium | 0,88 | 0,70 | 0,78 | 191 |

Beberapa catatan dari hasil di atas:

- Recall tertinggi ada di kelas Uveitis (0,86), sedangkan precision tertinggi di kelas Katarak (0,92).
- Recall Katarak paling rendah (0,63). Kelas ini memang paling sedikit datanya (593 gambar), meskipun sudah diberi class weight.
- Kesalahan yang paling sering terjadi adalah Pterygium yang diprediksi sebagai Uveitis (39 dari 191 gambar). Kemungkinan keduanya punya tampilan yang mirip bagi model.
- Akurasi validasi (70,84%) lebih rendah dari akurasi training (87,21%), jadi ada overfitting ringan.

## Cara kerja

Penelitian mengikuti tahapan CRISP-DM: business understanding, data understanding, data preparation, modeling, evaluation, dan deployment.

Alur di aplikasi:

```
Gambar (webcam / upload)
   -> praproses (RGB, resize 224x224, normalisasi)
   -> MobileNetV2 (transfer learning)
   -> prediksi + confidence + Grad-CAM
```

Arsitektur model:

- Base model MobileNetV2 dengan bobot ImageNet, 100 layer pertama dibekukan
- Di atasnya: GlobalAveragePooling2D, BatchNormalization, Dropout 0,4, Dense 128 (ReLU), Dropout 0,3, Dense 4 (Softmax)

Pengaturan training:

- Optimizer Adam (learning rate 0,0001), loss categorical crossentropy
- Maksimal 30 epoch, batch size 32
- Callback EarlyStopping dan ReduceLROnPlateau
- Augmentasi: rotasi, zoom, flip horizontal, dan pergeseran gambar
- Class weight untuk menangani ketidakseimbangan jumlah data antarkelas

## Dataset

Dataset berasal dari Roboflow Universe, proyek [klasifikasi-penyakit-mata versi 6](https://universe.roboflow.com/stardenburdenhardenbah/klasifikasi-penyakit-mata/dataset/6), dengan lisensi CC BY 4.0.

Data aslinya berformat YOLO (untuk object detection), jadi saya ubah ke format folder per kelas supaya bisa dipakai untuk klasifikasi. Split bawaan dataset juga tidak seimbang, terutama kelas Katarak. Karena itu semua data saya gabung lalu dibagi ulang per kelas dengan rasio 70:15:15.

| Kelas | Jumlah gambar |
|---|---|
| Uveitis | 1.255 |
| Katarak | 593 |
| Normal | 1.250 |
| Pterygium | 1.271 |
| Total | 4.369 |

## Teknologi

Python, TensorFlow/Keras, OpenCV, Scikit-learn, NumPy, Pillow, Streamlit, Google Colab, gdown.

## Menjalankan secara lokal

```bash
git clone https://github.com/Marufa14/deteksi_mata.git
cd deteksi_mata
pip install -r requirements.txt
streamlit run app.py
```

File model (`model_mata.h5`, sekitar 25 MB) tidak disimpan di repo ini. Saat pertama kali dijalankan, model diunduh otomatis dari Google Drive memakai `gdown`, jadi perlu koneksi internet. Setelah terunduh, file disimpan di folder lokal dan tidak diunduh lagi.

## Keterbatasan

- Akurasi 77% cukup untuk demo skrining awal, tapi belum layak untuk keperluan klinis.
- Hasil prediksi sangat bergantung pada cara pengambilan gambar. Foto mata dari dekat memberi hasil terbaik, sedangkan foto seluruh wajah bisa menghasilkan prediksi yang keliru.
- Kelas Katarak punya data paling sedikit dan recall paling rendah.
- Dataset bersifat publik dan tidak divalidasi secara klinis.

Rencana pengembangan: menambah data Katarak, menambahkan panduan posisi mata pada fitur webcam, regularisasi tambahan, dan versi yang bisa dijalankan offline.

## Disclaimer

Aplikasi ini adalah prototipe untuk penelitian dan edukasi, bukan alat medis. Hasilnya tidak menggantikan pemeriksaan dan diagnosis oleh dokter mata.


## Referensi

- Sandler, M. dkk. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
- Selvaraju, R. R. dkk. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.
- Dataset: stardenburdenhardenbah, Roboflow Universe (CC BY 4.0).
