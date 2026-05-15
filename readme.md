<h1 align="center">
🌿 CROP CONDITION ANALYSIS SYSTEM
</h1>

<h3 align="center">
Hệ thống nhận diện bệnh lá cây sử dụng YOLOv8 và Flask Web Application
</h3>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-WebApp-black?style=for-the-badge&logo=flask)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ObjectDetection-green?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-ComputerVision-red?style=for-the-badge&logo=opencv)
![Bootstrap](https://img.shields.io/badge/Bootstrap-Frontend-purple?style=for-the-badge&logo=bootstrap)

</div>

---

# 📖 1. Giới thiệu

Trong lĩnh vực nông nghiệp thông minh, việc phát hiện sớm bệnh trên lá cây đóng vai trò quan trọng trong việc giảm thiểu thiệt hại mùa vụ và nâng cao năng suất cây trồng.

Dự án **CROP CONDITION ANALYSIS SYSTEM** là một hệ thống ứng dụng trí tuệ nhân tạo giúp nhận diện bệnh lá cây thông qua mô hình học sâu **YOLOv8**, kết hợp với nền tảng web **Flask** để xây dựng giao diện trực quan, dễ sử dụng.

Hệ thống hỗ trợ:

- Nhận diện bệnh lá cây theo thời gian thực bằng camera (iVCam/Webcam)
- Tải ảnh lên để phân tích bệnh
- Tải video lên để phân tích từng frame
- Chụp ảnh thủ công
- Tự động lưu ảnh khi phát hiện bệnh nguy hiểm
- Lưu lịch sử phát hiện vào CSV
- Tải báo cáo kết quả

---

# 🎯 2. Các bệnh được hỗ trợ

Hệ thống hiện nhận diện 7 loại bệnh:

| STT | Tên mô hình | Tên tiếng Việt |
|----|------------|----------------|
| 1 | Anthracnose | Bệnh thán thư |
| 2 | Bacterial_Blight | Bệnh cháy lá do vi khuẩn |
| 3 | Deficiency Leaf | Thiếu dinh dưỡng ở lá |
| 4 | DryLeaf_Lemon | Khô lá cây chanh |
| 5 | Dry_Leaf-Lychee | Khô lá cây vải |
| 6 | Powdery_Mildew | Bệnh nấm phấn trắng |
| 7 | downey_mildew | Bệnh sương mai |

---

# 🛠️ 3. Công nghệ sử dụng

## Backend
- Python 3.11
- Flask

## AI / Computer Vision
- YOLOv8
- Ultralytics
- OpenCV
- Pandas

## Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript

## Môi trường phát triển
- Visual Studio Code
- Windows 10/11

---

# 📂 4. Cấu trúc project

```bash
Leaf_disease/
│
├── app.py
├── best.pt
├── history.csv
├── requirements.txt
├── README.md
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── upload_image.html
│   ├── upload_video.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── app.js
│   │
│   ├── uploads/
│   ├── outputs/
│   └── capture/
│
├── train/
├── valid/
└── test/
```

---

# 🚀 5. Chức năng chính

## 1. Nhận diện realtime
- Kết nối camera iVCam/Webcam
- Phân tích video trực tiếp
- Hiển thị bounding box bệnh

## 2. Upload ảnh
- Tải ảnh lá cây
- Phân tích bệnh
- Hiển thị kết quả trực tiếp

## 3. Upload video
- Phân tích video từng frame
- Xuất video đã detect

## 4. Chụp ảnh thủ công
- Chụp frame hiện tại
- Lưu ảnh kèm tên bệnh

Ví dụ:

```bash
Benh_than_thu_manual_20260515_101010.jpg
```

## 5. Chụp tự động
Khi phát hiện bệnh nguy hiểm:

```bash
CAO
```

hệ thống tự động lưu ảnh.

Ví dụ:

```bash
Benh_than_thu_auto_20260515_101010.jpg
```

## 6. Lưu lịch sử
Lưu vào:

```bash
history.csv
```

bao gồm:
- thời gian
- nguồn
- tên bệnh
- độ tin cậy
- mức độ nguy hiểm
- sức khỏe cây

---

# 🧠 6. Mô hình AI

Mô hình sử dụng:

```bash
YOLOv8
```

Dataset ban đầu:

```bash
13 classes
```

Sau khi lọc:

```bash
7 disease classes
```

Mô hình có xử lý:
- lọc class không phải bệnh
- giảm false prediction
- CLAHE preprocessing
- confidence threshold filtering

---

# ⚙️ 7. Cài đặt

## Clone project

```bash
git clone https://github.com/USERNAME/leaf-disease-web.git
```

## Di chuyển vào project

```bash
cd leaf-disease-web
```

## Cài thư viện

```bash
pip install -r requirements.txt
```

---

# ▶️ 8. Chạy project

```bash
python app.py
```

Mở trình duyệt:

```bash
http://127.0.0.1:5000
```

---

# 📸 9. Giao diện hệ thống

Có thể thêm ảnh chụp:

```bash
docs/dashboard.png
docs/upload.png
docs/history.png
```

---

# 📊 10. Dataset

Dataset gồm:

- Train set
- Validation set
- Test set

Số nhãn bệnh:

```bash
7
```

---

# 📬 11. Thông tin sinh viên

**Họ và tên:** Lò Đức Mạnh   
**Trường:** Dai Nam University

---

# © License

© 2026 AIoTLab, Faculty of Information Technology, DaiNam University. All rights reserved.
