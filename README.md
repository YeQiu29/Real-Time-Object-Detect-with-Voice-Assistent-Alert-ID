# Mata Digital: Asisten Penglihatan dengan Peringatan Suara

!

Proyek ini adalah sebuah sistem deteksi objek secara real-time yang dirancang sebagai alat bantu, khususnya bagi penyandang tunanetra. Aplikasi ini menggunakan kamera (webcam) untuk mengidentifikasi objek di lingkungan sekitar dan memberikan peringatan suara dalam Bahasa Indonesia, seperti "Awas ada 'kursi' di depanmu."

## ⚙️ Teknologi yang Digunakan
* **Python**: Bahasa pemrograman utama.
* **OpenCV**: Untuk memproses gambar dan video dari kamera.
* **YOLOv3 (You Only Look Once)**: Model Deep Learning yang digunakan untuk deteksi objek yang cepat dan akurat.
* **Pyttsx3**: Library untuk mengubah teks menjadi suara (Text-to-Speech).

## 🚀 Fitur Utama
* Deteksi objek secara langsung (real-time).
* Output suara yang jelas dalam Bahasa Indonesia untuk setiap objek yang terdeteksi.
* Menggunakan model COCO dataset yang mampu mengenali 80 jenis objek umum.
* Mudah dikembangkan untuk menambah lebih banyak objek dan pesan suara.

## 📋 Cara Menjalankan
1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/NAMA_USER_ANDA/NAMA_REPO_ANDA.git](https://github.com/NAMA_USER_ANDA/NAMA_REPO_ANDA.git)
    cd NAMA_REPO_ANDA
    ```
2.  **Install semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Jangan lupa untuk membuat file `requirements.txt` terlebih dahulu)*

3.  **Download file weight YOLOv3:**
    * Unduh file `yolov3.weights` dari [situs resmi YOLO](https://pjreddie.com/darknet/yolo/) dan letakkan di folder utama proyek.

4.  **Jalankan skrip utama:**
    ```bash
    python nama_file_anda.py
    ```
