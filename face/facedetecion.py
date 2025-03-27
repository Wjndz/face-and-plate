import cv2
import mediapipe as mp
import numpy as np
import os
import face_recognition
import pymongo
from datetime import datetime
import time
import uuid
import heapq

# 🔹 Kết nối MongoDB
client = pymongo.MongoClient("mongodb+srv://team2:team21234@cluster0.0tdjk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["face_db"]
collection = db["face_vectors"]

# 🔹 Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# 🔹 Thư mục lưu khuôn mặt
output_dir = "detected_faces"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 🔹 Mở camera (0: webcam mặc định, 1: iVCam)
cap = cv2.VideoCapture(0)

# Lưu trạng thái nhận diện
last_saved_time = 0
face_candidates = []  # Danh sách các khuôn mặt ứng viên (face_vector, confidence)
max_candidates = 3    # Số lượng ứng viên tối đa
min_confidence = 0.6  # Ngưỡng tin cậy tối thiểu

# 🟢 HÀM: Tìm user trong DB dựa vào vector khuôn mặt
def find_existing_user(face_vector):
    users = collection.find()
    for user in users:
        stored_vector = np.array(user["vector"])
        match = face_recognition.compare_faces([stored_vector], face_vector, tolerance=0.5)
        if match[0]:  
            return user["user_id"]  # Trả về user_id nếu khớp
    return None  # Không tìm thấy

def detect_face(frame):
    """
    Nhận diện khuôn mặt trong khung hình và trả về ID người dùng nếu tìm thấy
    """
    global last_saved_time, face_candidates
    
    # Đổi màu từ BGR (OpenCV) sang RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Chạy nhận diện khuôn mặt
    results = face_detection.process(rgb_frame)
    
    # Nếu phát hiện khuôn mặt
    if results.detections:
        for detection in results.detections:
            # Lấy confidence score
            confidence = detection.score[0]
            
            # Lấy tọa độ bounding box
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            
            # Vẽ bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Hiển thị confidence score
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Chỉ xử lý khuôn mặt có độ tin cậy trên ngưỡng
            if confidence > min_confidence:
                # Cắt khuôn mặt
                face_roi = frame[y:y+height, x:x+width]
                
                # Kiểm tra khuôn mặt có hợp lệ không
                if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                    continue
                
                # Chuyển thành dlib.rectangle để dễ sử dụng với face_recognition
                top, right, bottom, left = y, x + width, y + height, x
                
                # Tìm face encoding
                try:
                    face_encoding = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])[0]
                    
                    # Thêm vào danh sách ứng viên
                    heapq.heappush(face_candidates, (confidence, face_encoding))
                    
                    # Chỉ giữ max_candidates khuôn mặt có độ tin cậy cao nhất
                    if len(face_candidates) > max_candidates:
                        heapq.heappop(face_candidates)  # Loại bỏ khuôn mặt có độ tin cậy thấp nhất
                    
                    # Kiểm tra nếu đủ ứng viên và đã qua 2 giây kể từ lần lưu cuối cùng
                    current_time = time.time()
                    if len(face_candidates) >= max_candidates and (current_time - last_saved_time) > 2:
                        # Lấy khuôn mặt có độ tin cậy cao nhất
                        best_confidence, best_face_encoding = max(face_candidates)
                        
                        # Kiểm tra xem khuôn mặt này đã tồn tại trong DB chưa
                        existing_user_id = find_existing_user(best_face_encoding)
                        
                        if existing_user_id:
                            print(f"[EXISTING USER] Đã nhận diện user {existing_user_id}!")
                            
                            # Cập nhật thời gian truy cập
                            collection.update_one(
                                {"user_id": existing_user_id},
                                {"$set": {"last_access": current_time}}
                            )
                            
                            # Reset danh sách ứng viên
                            face_candidates = []
                            last_saved_time = current_time
                            
                            return existing_user_id
                        else:
                            # Tạo user mới chỉ khi chưa tồn tại
                            new_user_id = uuid.uuid4().hex[:8]  # Tạo ID ngắn gọn
                            
                            # Lưu ảnh khuôn mặt
                            timestamp = datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d_%H-%M-%S')
                            filename = f"{new_user_id}_{best_confidence:.2f}_{timestamp}.jpg"
                            filepath = os.path.join(output_dir, filename)
                            cv2.imwrite(filepath, face_roi)
                            
                            # Lưu vector khuôn mặt vào DB
                            collection.insert_one({
                                "user_id": new_user_id,
                                "vector": best_face_encoding.tolist(),
                                "created_at": current_time,
                                "last_access": current_time
                            })
                            
                            print(f"[NEW USER] Đã thêm user {new_user_id} vào database!")
                            print(f"[INFO] Lưu ảnh {filename} cho user {new_user_id}")
                            
                            # Reset danh sách ứng viên
                            face_candidates = []
                            last_saved_time = current_time
                            
                            return new_user_id
                except Exception as e:
                    print(f"[ERROR] {e}")
    
    # Hiển thị số lượng ứng viên đã thu thập
    cv2.putText(frame, f"Candidates: {len(face_candidates)}/{max_candidates}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return None

# Chạy chương trình nếu được gọi trực tiếp
if __name__ == "__main__":
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        user_id = detect_face(frame)
        if user_id:
            cv2.putText(frame, f"User: {user_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()