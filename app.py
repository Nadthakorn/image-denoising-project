import cv2
import threading
import time
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
# อนุญาตให้เว็บดึงข้อมูลข้ามโดเมนได้
CORS(app) 

# ==========================================
# ⚙️ ตั้งค่าแหล่งภาพ
# ==========================================
USE_CAMERA = False   # 🔴 วันซ้อมใช้ False (เล่นวิดีโอ), วันพรีเซนต์จริงเปลี่ยนเป็น True (ใช้กล้อง)
CAMERA_INDEX = 1     # 0 = กล้องโน้ตบุ๊ก, 1 หรือ 2 = กล้อง USB ที่นำมาต่อเพิ่ม
VIDEO_PATH = 'test_video.mp4'
ZONES_FILE = 'parking_zones.json' # ไฟล์สำหรับบันทึกพิกัดช่องจอด

# โครงสร้างข้อมูลตั้งต้น
buildings_data = [
    {"id": "A", "slots": []},
    {"id": "B", "slots": []},
    {"id": "C", "slots": []},
    {"id": "D", "slots": []},
    {"id": "E", "slots": []},
    {"id": "F", "slots": []},
    {"id": "G", "slots": []}
]

# ตัวแปรสำหรับระบบวาดช่องจอด
parking_zones = {}
edit_mode = False
drawing = False
temp_box = None

# ฟังก์ชันโหลดพิกัดที่เคยบันทึกไว้
def load_zones():
    global parking_zones
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE, 'r') as f:
            data = json.load(f)
            # แปลง key กลับเป็นตัวเลข
            parking_zones = {int(k): tuple(v) for k, v in data.items()}
            print("✅ โหลดพิกัดช่องจอดเรียบร้อยแล้ว")
    else:
        parking_zones = {}

# ฟังก์ชันบันทึกพิกัด
def save_zones():
    with open(ZONES_FILE, 'w') as f:
        json.dump(parking_zones, f)
    print("💾 บันทึกพิกัดช่องจอดลงไฟล์เรียบร้อยแล้ว!")

def check_intersection(car_box, zone_box):
    # เช็ค "จุดกึ่งกลางของรถ" ว่าตกลงไปในกรอบช่องจอดหรือไม่
    car_center_x = (car_box[0] + car_box[2]) / 2
    car_center_y = (car_box[1] + car_box[3]) / 2
    if zone_box[0] <= car_center_x <= zone_box[2] and zone_box[1] <= car_center_y <= zone_box[3]:
        return True
    return False

# ระบบจัดการเมาส์สำหรับวาด/ลบ กรอบ
def mouse_callback(event, x, y, flags, param):
    global drawing, temp_box, parking_zones, edit_mode

    if not edit_mode:
        return

    # คลิกซ้ายค้างเพื่อเริ่มวาด
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        temp_box = [x, y, x, y]

    # ลากเมาส์
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing and temp_box is not None:
            temp_box[2], temp_box[3] = x, y

    # ปล่อยคลิกซ้ายเพื่อสิ้นสุดการวาด
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if temp_box is not None:
            x1, y1 = min(temp_box[0], temp_box[2]), min(temp_box[1], temp_box[3])
            x2, y2 = max(temp_box[0], temp_box[2]), max(temp_box[1], temp_box[3])
            if x2 - x1 > 20 and y2 - y1 > 20: # ป้องกันการเผลอคลิกจุดเล็กๆ
                new_id = len(parking_zones)
                parking_zones[new_id] = (x1, y1, x2, y2)
            temp_box = None

    # คลิกขวาเพื่อลบช่องจอดที่คลิก
    elif event == cv2.EVENT_RBUTTONDOWN:
        to_delete = None
        for idx, box in parking_zones.items():
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                to_delete = idx
                break
        if to_delete is not None:
            del parking_zones[to_delete]
            # จัดเรียง ID ใหม่ให้ต่อเนื่อง
            reindexed = {i: v for i, v in enumerate(parking_zones.values())}
            parking_zones.clear()
            parking_zones.update(reindexed)

def run_yolo():
    global edit_mode
    model = YOLO('yolov8n.pt')  
    load_zones() # โหลดพิกัดตอนเริ่มโปรแกรม
    
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"🎥 กำลังเปิดกล้อง (Index {CAMERA_INDEX})...")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"🎬 กำลังเล่นวิดีโอจำลอง...")
        
    cv2.namedWindow("KKU Parking Camera")
    cv2.setMouseCallback("KKU Parking Camera", mouse_callback)
    
    while True:
        success, frame = cap.read()
        if not success:
            if not USE_CAMERA:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
                
        results = model(frame, classes=[2, 3, 5, 7]) # ตรวจเฉพาะยานพาหนะ
        building_a = next(b for b in buildings_data if b["id"] == "A")
        
        # กำหนดสถานะชั่วคราวให้เท่ากับจำนวนช่องจอดที่มีอยู่จริง
        num_zones = len(parking_zones)
        slot_occupied = [False] * num_zones if num_zones > 0 else []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_box = (x1, y1, x2, y2)
                
                # วาดกรอบรถสีฟ้า (ซ่อนกรอบรถชั่วคราวถ้าอยู่ในโหมดวาด เพื่อไม่ให้ลายตา)
                if not edit_mode:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                for idx, zone_box in parking_zones.items():
                    if check_intersection(car_box, zone_box):
                        slot_occupied[idx] = True
                        
        # อัปเดตข้อมูลส่งให้หน้าเว็บ
        building_a["slots"] = [not occ for occ in slot_occupied]

        # วาดกรอบช่องจอดลงบนภาพ
        for idx, zone_box in parking_zones.items():
            is_occupied = slot_occupied[idx] if idx < len(slot_occupied) else False
            color = (0, 0, 255) if is_occupied else (0, 255, 0)
            # ถ้าอยู่ในโหมดแก้ไข ให้กรอบเป็นสีเหลืองเพื่อแสดงว่าพร้อมแก้ไข
            if edit_mode:
                color = (0, 255, 255) 
                
            cv2.rectangle(frame, (zone_box[0], zone_box[1]), (zone_box[2], zone_box[3]), color, 2)
            cv2.putText(frame, f"A-{idx+1:02d}", (zone_box[0], zone_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # วาดกรอบชั่วคราวขณะกำลังลากเมาส์
        if drawing and temp_box is not None:
            cv2.rectangle(frame, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), (255, 0, 255), 2)

        # แสดงเมนูคำแนะนำบนหน้าจอ
        if edit_mode:
            cv2.putText(frame, "EDIT MODE: Drag to draw | Right-Click to delete", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'S' to Save | 'C' to Clear | 'E' to Exit Edit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Press 'E' to Edit Slots | 'Q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("KKU Parking Camera", frame)
        
        # ระบบปุ่มคีย์บอร์ด
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('e') or key == ord('E'):
            edit_mode = not edit_mode # เปิด/ปิด โหมดแก้ไข
        elif key == ord('s') or key == ord('S'):
            if edit_mode: save_zones() # บันทึก
        elif key == ord('c') or key == ord('C'):
            if edit_mode: 
                parking_zones.clear() # ล้างทั้งหมด

    cap.release()
    cv2.destroyAllWindows()

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"buildings": buildings_data})

if __name__ == '__main__':
    t = threading.Thread(target=run_yolo)
    t.daemon = True
    t.start()
    
    print("🚀 ระบบ API รันแล้วที่ http://localhost:3000/api/status")
    app.run(host='0.0.0.0', port=3000)
