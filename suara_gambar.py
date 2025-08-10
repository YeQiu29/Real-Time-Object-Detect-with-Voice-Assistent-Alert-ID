import cv2
import numpy as np
import pyttsx3  # Import library untuk text-to-speech

# Inisialisasi text-to-speech
engine = pyttsx3.init()

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

# Memuat gambar
img = cv2.imread("room_ser.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)  # Mengubah ukuran gambar
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
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)  # Menggambar kotak
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)  # Menambahkan label

        # Mengeluarkan suara untuk objek yang terdeteksi
        text_to_speak = f"{label} terdeteksi"
        engine.say(text_to_speak)  # Mengatur teks untuk diucapkan
        engine.runAndWait()  # Menjalankan perintah suara

# Menampilkan gambar dengan hasil deteksi
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
