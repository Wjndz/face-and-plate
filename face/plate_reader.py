import cv2
from easyocr import Reader
from PIL import ImageFont, ImageDraw, Image
import numpy as np
from pymongo import MongoClient
import re
import time
from collections import Counter

# Kết nối MongoDB
client = MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]  
collection = db["face_vectors"]  

# Cấu hình EasyOCR
reader = Reader(['en', 'vi'])

# Cấu hình camera
cap = cv2.VideoCapture(1)  
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Font vẽ chữ
fontpath = "./arial.ttf"
font = ImageFont.truetype(fontpath, 32)
b, g, r, a = 0, 255, 0, 0  

# Các biến cho việc theo dõi kết quả OCR
frame_count = 0
ocr_results = []
scanning = True  # Biến kiểm soát quét OCR

# Các biến cho việc thu thập nhiều kết quả OCR
candidate_plates = []      # Danh sách các biển số ứng viên
confidence_threshold = 0.6 # Ngưỡng tin cậy tối thiểu
max_candidates = 10        # Số lượng ứng viên tối đa trước khi quyết định
min_candidates = 5         # Số lượng ứng viên tối thiểu trước khi quyết định
last_detection_time = 0    # Thời gian phát hiện biển số cuối cùng

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

# Kích thước vùng quét
zone_w, zone_h = 300, 200

# Hàm chuẩn hóa biển số xe
def normalize_plate(text):
    # Loại bỏ các ký tự đặc biệt không liên quan
    text = re.sub(r'[^\w\s\.-]', '', text)
    
    # Bảng kiểm tra và thay thế cho các ký tự dễ nhầm lẫn
    confusion_map = {
        '8': 'B',  # Trong vị trí ký tự của biển số, 8 thường là B
        '0': 'D',  # 0 thường là D
        'O': 'D',  # O thường là D
    
        '5': 'S',  # 5 thường là S
        'G': '6',  # G thường là 6 khi ở vị trí số
    }
    
    # Xử lý biển số xe có khoảng trắng
    parts = text.split(' ', 1)
    if len(parts) == 2:
        prefix, suffix = parts
        
        # Phân tách tiền tố thành phần số đầu và phần chữ
        prefix_match = re.match(r'(\d{2})-?([A-Z0-9]+)', prefix)
        
        if prefix_match:
            num_part, char_part = prefix_match.groups()
            
            # Xử lý phần chữ cái (thay 8 -> B, v.v.)
            for i, char in enumerate(char_part):
                if char in confusion_map and char.isdigit():  # Nếu là số ở vị trí chữ cái
                    char_part = char_part[:i] + confusion_map[char] + char_part[i+1:]
            
            # Tạo lại tiền tố với dấu gạch ngang
            prefix = f"{num_part}-{char_part}"
        
        # Kiểm tra và định dạng phần hậu tố
        if suffix.isdigit() and len(suffix) == 5:
            # Định dạng số 5 chữ số thành format xxx.xx
            suffix = f"{suffix[:3]}.{suffix[3:]}"
        
        text = f"{prefix} {suffix}"
    
    # Thêm dấu "-" nếu thiếu
    text = re.sub(r'(\d{2})([A-Z])', r'\1-\2', text)
    
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Hàm kiểm tra tính hợp lệ của biển số xe Việt Nam
def is_valid_vietnamese_plate(plate):
    """Kiểm tra xem biển số xe có tuân theo định dạng biển số Việt Nam không"""
    # Kiểm tra các mẫu biển số xe phổ biến tại Việt Nam
    patterns = [
        # Biển số xe ô tô: 2 số đầu + 1 chữ + 4-5 số (vd: 51G-12345)
        r'^\d{2}-[A-Z]\d{4,5}$',
        # Biển số xe ô tô có 2 chữ cái: 2 số đầu + 2 chữ + 3-5 số (vd: 30AB-1234)
        r'^\d{2}-[A-Z]{2}\d{3,5}$',
        # Biển số xe máy: 2 số đầu + 1 chữ + 1 số + dấu chấm + 2 số (vd: 59H-2.12)
        r'^\d{2}-[A-Z]\d\.\d{2}$',
        # Biển số xe máy có 2 chữ cái: 2 số đầu + 2 chữ + 1 số + dấu chấm + 2 số (vd: 59HA-2.12)
        r'^\d{2}-[A-Z]{2}\d\.\d{2}$',
        # Biển số xe ô tô kiểu mới với format: 2 số đầu + 1-2 chữ cái + khoảng trắng + 3-5 số (vd: 12-B1 168.88)
        r'^\d{2}-[A-Z][A-Z0-9]+ \d{3}(\.\d{2})?$',
    ]
    
    for pattern in patterns:
        if re.match(pattern, plate):
            return True
            
    return False

# Hàm ghép biển số từ nhiều dòng
def combine_plate_lines(text_lines):
    # Nếu không có kết quả, trả về chuỗi rỗng
    if not text_lines:
        return ""
        
    # Sắp xếp các dòng theo tọa độ y tăng dần (từ trên xuống dưới)
    sorted_lines = sorted(text_lines, key=lambda x: x[0][0][1])
    
    # Lấy text từ các dòng đã sắp xếp
    texts = [line[1] for line in sorted_lines]
    
    # Nếu có 2 dòng trở lên, ghép dòng đầu và dòng thứ hai
    if len(texts) >= 2:
        top_text = texts[0].strip()
        bottom_text = texts[1].strip()
        
        # Ghép 2 dòng với khoảng trắng
        combined = f"{top_text} {bottom_text}"
        return combined
    elif len(texts) == 1:
        return texts[0].strip()
    else:
        return ""

# Hàm tiền xử lý ảnh để cải thiện OCR
def preprocess_image(image):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Giảm nhiễu với GaussianBlur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Áp dụng ngưỡng thích ứng
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Áp dụng các phép biến đổi hình thái học
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Tăng độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    
    return opening, enhanced

# Hàm chọn biển số xe đáng tin cậy nhất
def select_best_plate(candidates):
    if not candidates:
        return None
    
    # Đếm số lần xuất hiện của mỗi biển số
    counter = Counter(candidates)
    
    # Sắp xếp theo số lần xuất hiện (cao -> thấp)
    sorted_plates = counter.most_common()
    
    # Ưu tiên biển số hợp lệ theo định dạng Việt Nam
    for plate, count in sorted_plates:
        # Nếu biển số hợp lệ và xuất hiện ít nhất 3 lần
        if is_valid_vietnamese_plate(plate) and count >= 3:
            print(f"🏁 Đã chọn biển số: {plate} (xuất hiện {count}/{len(candidates)} lần)")
            return plate
    
    # Nếu không tìm thấy biển số hợp lệ, chọn biển số xuất hiện nhiều nhất
    best_plate, count = sorted_plates[0]
    
    # Nếu biển số xuất hiện quá ít, trả về None
    if count < 3:  # Yêu cầu tối thiểu 3 lần phát hiện giống nhau
        return None
    
    print(f"🏁 Đã chọn biển số: {best_plate} (xuất hiện {count}/{len(candidates)} lần)")
    return best_plate

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ camera!")
        break

    frame_count += 1
    h_img, w_img = frame.shape[:2]

    # Xác định vùng OCR ở giữa màn hình
    zone_x = (w_img - zone_w) // 2
    zone_y = (h_img - zone_h) // 2

    # Vẽ khung vùng quét
    cv2.rectangle(frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), (0, 255, 0), 2)

    # Chuyển frame sang PIL để vẽ chữ
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)

    # Kiểm tra nếu đã có biển số hợp lệ, thì dừng quét
    if not scanning:
        draw.text((10, 10), "Đã dừng quét. Chờ người dùng mới...", font=font, fill=(r, 0, 0, a))
    else:
        # Chạy OCR mỗi 5 frame để tăng tốc độ
        if frame_count % 5 == 0:
            # Trích xuất vùng ROI để OCR (chỉ xử lý vùng quan tâm)
            roi = frame[zone_y:zone_y+zone_h, zone_x:zone_x+zone_w]
            
            if roi.size == 0:
                continue
                
            # Tiền xử lý ảnh để cải thiện OCR
            processed_roi, enhanced_roi = preprocess_image(roi)
            
            # Hiển thị ảnh tiền xử lý
            cv2.imshow("Preprocessed", processed_roi)
            cv2.imshow("Enhanced", enhanced_roi)
            
            # Thực hiện OCR trên các phiên bản khác nhau của ảnh
            results1 = reader.readtext(roi, detail=1)
            results2 = reader.readtext(processed_roi, detail=1)
            results3 = reader.readtext(enhanced_roi, detail=1)
            
            # Kết hợp kết quả, ưu tiên kết quả có độ tin cậy cao
            ocr_results = results1 + results2 + results3
            
            # Lọc kết quả, chỉ giữ lại các kết quả có độ tin cậy cao
            ocr_results = [r for r in ocr_results if r[2] > confidence_threshold]
            
            # Sắp xếp theo độ tin cậy giảm dần
            ocr_results.sort(key=lambda x: x[2], reverse=True)
            
            # Vẽ các bbox và text lên ảnh ROI
            roi_with_boxes = roi.copy()
            for i, (bbox, text, prob) in enumerate(ocr_results[:3]):  # Chỉ hiển thị top 3 kết quả
                # Lấy tọa độ bbox
                topleft = (int(bbox[0][0]), int(bbox[0][1]))
                bottomright = (int(bbox[2][0]), int(bbox[2][1]))
                
                # Vẽ bbox
                cv2.rectangle(roi_with_boxes, topleft, bottomright, (0, 255, 0), 2)
                
                # Hiển thị text và độ tin cậy bên cạnh bbox
                cv2.putText(roi_with_boxes, f"{text} ({prob:.2f})", 
                           (topleft[0], topleft[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Hiển thị ROI với các bbox
            cv2.imshow("ROI with OCR", roi_with_boxes)
        
        # Ghép các dòng thành biển số hoàn chỉnh
        combined_plate = combine_plate_lines(ocr_results)
        
        # Chuẩn hóa biển số
        normalized_plate = normalize_plate(combined_plate)
        
        # Hiển thị biển số trên màn hình
        if normalized_plate:
            draw.text((zone_x + 10, zone_y - 40), normalized_plate, font=font, fill=(b, g, r, a))
            
            # Thêm vào danh sách ứng viên
            candidate_plates.append(normalized_plate)
            last_detection_time = time.time()
            
            # Hiển thị số lượng ứng viên đã thu thập
            draw.text((10, h_img - 40), f"Đã có: {len(candidate_plates)}/{max_candidates} ứng viên", 
                     font=font, fill=(0, g, 0, a))
        
        # Kiểm tra nếu đã thu thập đủ số lượng ứng viên hoặc đã hết thời gian
        current_time = time.time()
        timeout_condition = (current_time - last_detection_time > 3.0 and len(candidate_plates) >= min_candidates)
        max_candidates_condition = len(candidate_plates) >= max_candidates
        
        if (timeout_condition or max_candidates_condition) and candidate_plates:
            # Chọn biển số đáng tin cậy nhất
            best_plate = select_best_plate(candidate_plates)
            
            if best_plate:
                # Kiểm tra biển số hợp lệ theo định dạng Việt Nam
                if is_valid_vietnamese_plate(best_plate):
                    # Tìm user mới nhất chưa có biển số xe
                    last_entry = collection.find_one({"license_plate": {"$exists": False}}, sort=[("created_at", -1)])
                    
                    if last_entry:
                        # Cập nhật biển số cho user này
                        collection.update_one({"_id": last_entry["_id"]}, {"$set": {"license_plate": best_plate}})
                        print(f"✅ Đã cập nhật biển số '{best_plate}' cho user_id: {last_entry['user_id']}")
                        
                        # In thông tin user có biển số xe
                        user_info = collection.find_one({"_id": last_entry["_id"]})
                        print(f"📊 Thông tin đầy đủ:")
                        print(f"  👤 User ID: {user_info['user_id']}")
                        print(f"  🚗 Biển số: {user_info['license_plate']}")
                        print(f"  🕒 Thời gian tạo: {user_info.get('created_at', 'N/A')}")
                        
                        # Dừng quét sau khi cập nhật thành công
                        scanning = False
                    else:
                        print(f"⚠️ Không tìm thấy user mới nào chưa có biển số xe trong database!")
                else:
                    print(f"⚠️ Biển số '{best_plate}' không hợp lệ theo định dạng Việt Nam, bỏ qua!")
            
            # Reset danh sách ứng viên
            candidate_plates = []

    # Chuyển ảnh PIL về OpenCV để hiển thị
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("License Plate Scanner", frame)

    # Kiểm tra có user mới được thêm vào database không
    new_user = collection.find_one({"license_plate": {"$exists": False}}, sort=[("created_at", -1)])
    if new_user and not scanning:
        print(f"🔄 Phát hiện user_id mới: {new_user['user_id']} vừa được thêm vào database, tiếp tục quét...")
        print(f"  🕒 Thời gian tạo: {new_user.get('created_at', 'N/A')}")
        scanning = True  # Bật lại OCR nếu có user mới
        candidate_plates = []  # Reset danh sách ứng viên

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()