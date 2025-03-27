import cv2
import mediapipe as mp
import numpy as np
import face_recognition
from easyocr import Reader
from PIL import ImageFont, ImageDraw, Image
import re
from collections import Counter
import time
from pymongo import MongoClient

# ---------------------------
# 1) KẾT NỐI DB
# ---------------------------
client = MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["face_vectors"]

# ---------------------------
# 2) CẤU HÌNH OCR
# ---------------------------
reader = Reader(['en', 'vi'])

# ---------------------------
# 3) MỞ 2 CAMERA
# ---------------------------
# Camera dành cho nhận diện khuôn mặt
cap_face = cv2.VideoCapture(0)
cap_face.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_face.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Camera dành cho nhận diện biển số
cap_plate = cv2.VideoCapture(1)
cap_plate.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_plate.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------------------------
# 4) MEDIAPIPE FACE DETECTION
# ---------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.6
)

# ---------------------------
# 5) FONT & MÀU VẼ
# ---------------------------
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b, g, r, a = 0, 255, 0, 0  # Xanh lá

# ---------------------------
# BIẾN THEO DÕI
# ---------------------------
frame_count = 0
candidate_plates = []
confidence_threshold = 0.6
max_candidates = 10
min_candidates = 5
last_detection_time = 0

# ROI cho OCR biển số (ở giữa frame_plate)
zone_w, zone_h = 300, 200

# Kết quả nhận diện
found_id = None   # user_id (khuôn mặt)
best_plate = None # kết quả OCR biển số

# -------------------------------------------------
# CÁC HÀM XỬ LÝ BIỂN SỐ (GIỮ NGUYÊN NHƯ BẠN ĐÃ DÙNG)
# -------------------------------------------------
def normalize_plate(text):
    text = re.sub(r'[^\w\s\.-]', '', text)
    confusion_map = {
        '8': 'B',
        '0': 'D',
        'O': 'D',
        '5': 'S',
        'G': '6',
    }
    parts = text.split(' ', 1)
    if len(parts) == 2:
        prefix, suffix = parts
        prefix_match = re.match(r'(\d{2})-?([A-Z0-9]+)', prefix)
        if prefix_match:
            num_part, char_part = prefix_match.groups()
            for i, char in enumerate(char_part):
                if char in confusion_map and char.isdigit():
                    char_part = char_part[:i] + confusion_map[char] + char_part[i+1:]
            prefix = f"{num_part}-{char_part}"
        if suffix.isdigit() and len(suffix) == 5:
            suffix = f"{suffix[:3]}.{suffix[3:]}"
        text = f"{prefix} {suffix}"
    text = re.sub(r'(\d{2})([A-Z])', r'\1-\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_vietnamese_plate(plate):
    patterns = [
        r'^\d{2}-[A-Z]\d{4,5}$',
        r'^\d{2}-[A-Z]{2}\d{3,5}$',
        r'^\d{2}-[A-Z]\d\.\d{2}$',
        r'^\d{2}-[A-Z]{2}\d\.\d{2}$',
        r'^\d{2}-[A-Z][A-Z0-9]+ \d{3}(\.\d{2})?$',
    ]
    for pattern in patterns:
        if re.match(pattern, plate):
            return True
    return False

def combine_plate_lines(text_lines):
    if not text_lines:
        return ""
    sorted_lines = sorted(text_lines, key=lambda x: x[0][0][1])
    texts = [line[1] for line in sorted_lines]
    if len(texts) >= 2:
        return f"{texts[0].strip()} {texts[1].strip()}"
    elif len(texts) == 1:
        return texts[0].strip()
    return ""

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    return opening, enhanced

from collections import Counter
def select_best_plate(candidates):
    if not candidates:
        return None
    counter = Counter(candidates)
    sorted_plates = counter.most_common()
    for plate, count in sorted_plates:
        if is_valid_vietnamese_plate(plate) and count >= 3:
            print(f"🏁 Đã chọn biển số: {plate} (xuất hiện {count}/{len(candidates)} lần)")
            return plate
    best_plate, count = sorted_plates[0]
    if count < 3:
        return None
    print(f"🏁 Đã chọn biển số: {best_plate} (xuất hiện {count}/{len(candidates)} lần)")
    return best_plate

# ---------------------------
# HÀM TÌM USER TRONG DB BẰNG VECTOR
# ---------------------------
def find_existing_user(face_vector):
    users = collection.find()
    for user in users:
        stored_vector = np.array(user["vector"])
        match = face_recognition.compare_faces([stored_vector], face_vector, tolerance=0.5)
        if match[0]:
            return user["user_id"]
    return None

# ---------------------------
# CHECKOUT: So sánh user_id & biển số
# ---------------------------
def perform_checkout(found_id, best_plate):
    user = collection.find_one({
        "user_id": found_id,
        "license_plate": best_plate
    })
    if user:
        checkout_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[CHECKOUT] Thành công! user_id={found_id}, biển số={best_plate}")
        db["logs"].insert_one({
            "user_id": found_id,
            "license_plate": best_plate,
            "action": "checkout",
            "timestamp": checkout_time
        })
        # Xóa thông tin user sau khi checkout
        collection.delete_one({"user_id": found_id})
    else:
        print(f"[CHECKOUT] Thất bại! Không khớp user_id='{found_id}' với biển số='{best_plate}'")

# ---------------------------
# VÒNG LẶP CHÍNH
# ---------------------------
while True:
    # ĐỌC FRAME TỪ 2 CAMERA
    ret_face, frame_face = cap_face.read()
    ret_plate, frame_plate = cap_plate.read()

    if not ret_face or not ret_plate:
        print("Không thể đọc khung hình từ 1 trong 2 camera!")
        break

    frame_count += 1

    # ============== PHẦN A: NHẬN DIỆN KHUÔN MẶT (CAM FACE) ==============
    h_face, w_face = frame_face.shape[:2]
    rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(rgb_face)
    if face_results and face_results.detections:
        for detection in face_results.detections:
            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * w_face)
            y = int(box.ymin * h_face)
            w_box = int(box.width * w_face)
            h_box = int(box.height * h_face)

            # Vẽ khung
            cv2.rectangle(frame_face, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)

            # Cắt khuôn mặt
            face_img = frame_face[y:y + h_box, x:x + w_box]
            if face_img.size != 0:
                small_face = cv2.resize(face_img, (150, 150))
                face_encoding = face_recognition.face_encodings(small_face)
                if face_encoding:
                    face_vector = face_encoding[0]
                    user_id = find_existing_user(face_vector)
                    if user_id:
                        found_id = user_id
                        cv2.putText(frame_face, f"User: {found_id}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame_face, "Unknown",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 2)

    # ============== PHẦN B: NHẬN DIỆN BIỂN SỐ (CAM PLATE) ==============
    h_plate, w_plate = frame_plate.shape[:2]
    zone_x = (w_plate - zone_w) // 2
    zone_y = (h_plate - zone_h) // 2
    cv2.rectangle(frame_plate, (zone_x, zone_y),
                  (zone_x + zone_w, zone_y + zone_h),
                  (0, 255, 0), 2)

    # Mỗi 5 frame, chạy OCR
    if frame_count % 5 == 0:
        roi = frame_plate[zone_y:zone_y + zone_h, zone_x:zone_x + zone_w]
        if roi.size != 0:
            processed_roi, enhanced_roi = preprocess_image(roi)
            cv2.imshow("Preprocessed", processed_roi)
            cv2.imshow("Enhanced", enhanced_roi)

            results1 = reader.readtext(roi, detail=1)
            results2 = reader.readtext(processed_roi, detail=1)
            results3 = reader.readtext(enhanced_roi, detail=1)
            ocr_results = results1 + results2 + results3

            # Lọc theo độ tin cậy
            ocr_results = [r for r in ocr_results if r[2] > confidence_threshold]
            ocr_results.sort(key=lambda x: x[2], reverse=True)

            # Vẽ bbox OCR trên ROI
            roi_with_boxes = roi.copy()
            for i, (bbox, text, prob) in enumerate(ocr_results[:3]):
                topleft = (int(bbox[0][0]), int(bbox[0][1]))
                bottomright = (int(bbox[2][0]), int(bbox[2][1]))
                cv2.rectangle(roi_with_boxes, topleft, bottomright, (0,255,0), 2)
                cv2.putText(roi_with_boxes, f"{text} ({prob:.2f})",
                            (topleft[0], topleft[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow("ROI with OCR", roi_with_boxes)

            combined_plate = combine_plate_lines(ocr_results)
            normalized_plate = normalize_plate(combined_plate)
            if normalized_plate:
                best_plate = normalized_plate
                cv2.putText(frame_plate, f"Plate: {best_plate}",
                            (10, h_plate - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                candidate_plates.append(normalized_plate)
                last_detection_time = time.time()

    # ============== CHỌN BIỂN SỐ TỐT NHẤT ==============
    current_time = time.time()
    timeout_condition = (current_time - last_detection_time > 3.0) and (len(candidate_plates) >= min_candidates)
    max_candidates_condition = len(candidate_plates) >= max_candidates
    if (timeout_condition or max_candidates_condition) and candidate_plates:
        best_plate_tmp = select_best_plate(candidate_plates)
        if best_plate_tmp:
            best_plate = best_plate_tmp
            if is_valid_vietnamese_plate(best_plate):
                print(f"[INFO] Biển số tốt nhất: {best_plate}")
            else:
                print(f"⚠ Biển số '{best_plate}' không hợp lệ!")
        candidate_plates = []

    # ============== CHECKOUT NẾU ĐỦ THÔNG TIN ==============
    if found_id and best_plate:
        perform_checkout(found_id, best_plate)
        found_id = None
        best_plate = None
        candidate_plates = []

    # Hiển thị 2 cửa sổ
    cv2.imshow("Face Recognition", frame_face)
    cv2.imshow("License Plate Recognition", frame_plate)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_face.release()
cap_plate.release()
cv2.destroyAllWindows()
