import cv2
import numpy as np
import pyttsx3  # Import library untuk text-to-speech

# Inisialisasi text-to-speech
engine = pyttsx3.init()

# Mengatur suara ke bahasa Indonesia
voices = engine.getProperty('voices')
for voice in voices:
    if 'ind' in voice.languages:  # Mencari suara yang mendukung bahasa Indonesia
        engine.setProperty('voice', voice.id)
        break

# Memuat model YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Memuat kelas objek dari file coco.names
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Mendapatkan nama layer dari model
layer_names = net.getLayerNames()
out_layers = net.getUnconnectedOutLayers()

# Menangani format output yang berbeda
if isinstance(out_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in out_layers.flatten()]
else:
    output_layers = [layer_names[out_layers - 1]]

# Menghasilkan warna acak untuk setiap kelas
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Kamus untuk output suara dalam bahasa Indonesia
output_messages = {
    "person": "Awas ada 'orang' di depanmu.",
    "chair": "Awas ada 'kursi' di depanmu.",
    "laptop": "Awas ada 'laptop' di depanmu.",
    "bottle": "Awas ada 'botol' di depanmu.",
    "dog": "Awas ada 'anjing' di depanmu.",
    "cat": "Awas ada 'kucing' di depanmu.",
    # Tambahkan lebih banyak objek sesuai kebutuhan
}

# Menggunakan kamera
cap = cv2.VideoCapture(0)  # 0 untuk kamera default

while True:
    # Membaca frame dari kamera
    ret, img = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Mengubah ukuran gambar untuk memperbesar tampilan
    img = cv2.resize(img, (1280, 720))  # Mengubah ukuran gambar menjadi 1280x720
    height, width, channels = img.shape

    # Mendeteksi objek
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Menampilkan informasi pada layar
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Ambang batas kepercayaan
                # Objek terdeteksi
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Koordinat persegi panjang
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Menggunakan Non-Maximum Suppression untuk menghapus kotak yang tumpang tindih
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Menampilkan hasil deteksi pada gambar
    font = cv2.FONT_HERSHEY_PLAIN
    detected_objects = set()  # Set untuk menyimpan objek yang sudah diucapkan
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Menggambar kotak
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)  # Menambahkan label

            # Mengeluarkan suara untuk objek yang terdeteksi
            if label in output_messages and label not in detected_objects:  # Cek jika objek ada di kamus dan belum diucapkan
                text_to_speak = output_messages[label]  # Ambil pesan dari kamus
                engine.say(text_to_speak)  # Mengatur teks untuk diucapkan
                engine.runAndWait()  # Menjalankan perintah suara
                detected_objects.add(label)  # Tambahkan objek ke set

    # Menampilkan gambar dengan hasil deteksi
    cv2.imshow("Deteksi Objek", img)

    # Mengubah ukuran jendela tampilan
    cv2.resizeWindow("Deteksi Objek", 1280, 720)  # Mengatur ukuran jendela menjadi 1280x720

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()

# Gasss