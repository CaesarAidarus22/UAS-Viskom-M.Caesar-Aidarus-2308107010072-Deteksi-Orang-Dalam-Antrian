Sistem Deteksi dan Penghitungan Orang pada Foto Antrian

UAS Mata Kuliah Visi Komputer – Universitas Syiah Kuala (USK)

Proyek ini merupakan implementasi Computer Vision untuk mendeteksi dan menghitung jumlah orang pada foto antrian menggunakan model YOLOv3-tiny melalui OpenCV DNN. Sistem melakukan preprocessing, deteksi objek, perhitungan jumlah orang, klasifikasi kepadatan antrian, serta menampilkan hasil deteksi secara visual.

⸻

📌 Fitur Utama
	•	Deteksi manusia berdasarkan model CNN YOLOv3-tiny
	•	Perhitungan jumlah orang dalam foto
	•	Pengkategorian tingkat kepadatan antrian:
	•	Sepi (0–2 orang)
	•	Sedang (3–5 orang)
	•	Ramai (>5 orang)
	•	Preprocessing (Grayscale, Gaussian Blur, Edge Detection)
	•	Filtering heuristik untuk mengurangi false positive
	•	Overlay bounding box & teks seperti Augmented Reality

⸻

📂 Struktur Folder

uas-viskom-foto/
│
├── images/
│   └── input.jpg
│
├── models/
│   ├── yolov3-tiny.cfg
│   └── yolov3-tiny.weights
│
├── output/
│   └── hasil.jpg
│
├── main.py
├── detect.py
├── preprocess.py
└── requirements.txt


⸻

⚙️ Instalasi

1. Clone Repository

git clone https://github.com/CaesarAidarus22/UAS-Viskom-M.Caesar-Aidarus-2308107010072-Deteksi-Orang-Dalam-Antrian.git
cd UAS-Viskom-M.Caesar-Aidarus-2308107010072-Deteksi-Orang-Dalam-Antrian

2. Buat Virtual Environment (Opsional)

python3 -m venv viskom-venv
source viskom-venv/bin/activate

3. Install Dependency

pip install -r requirements.txt


⸻

▶️ Cara Menjalankan Program

Letakkan foto input ke folder images, misalnya:

images/input.jpg

Lalu jalankan:

python main.py --image images/input.jpg

Output akan tersimpan di:

output/hasil.jpg


⸻

📊 Contoh Hasil

🧠 Penjelasan Teknis Singkat

1. Preprocessing
	•	cv2.cvtColor()
	•	GaussianBlur()
	•	Canny()

2. Deteksi CNN (YOLOv3-tiny)
	•	Blob extraction
	•	Forward pass OpenCV DNN
	•	NMS (Non-Maximum Suppression)

3. Filtering Heuristik
	•	Minimum bounding box area
	•	Rasio tinggi-lebar untuk memfilter tas/koper

4. Output
	•	Bounding box
	•	Jumlah orang
	•	Status (Sepi/Sedang/Ramai)

⸻

🚧 Keterbatasan
	•	Orang yang saling menutupi (occlusion) mungkin terlewat
	•	Orang sangat kecil sulit dideteksi
	•	Model YOLOv3-tiny kurang akurat dibanding YOLOv5/YOLOv8

⸻

🚀 Pengembangan Selanjutnya
	•	Upgrade ke YOLOv8
	•	Fine-tuning dengan dataset antrian
	•	Pembuatan GUI atau versi video realtime

⸻

Dibuat oleh

Nama: M. Caesar Aidarus
NIM: 2308107010072
Mata Kuliah: Visi Komputer 
Tahun: 2025