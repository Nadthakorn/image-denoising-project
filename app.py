import cv2
import threading
import time
import json
import os
from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# ==========================================
# ⚙️ 1. ตั้งค่าพื้นฐานสำหรับเซิร์ฟเวอร์ (Flask)
# ==========================================
app = Flask(__name__)
CORS(app) # อนุญาตให้เว็บข้ามโดเมนดึงข้อมูลได้ (ป้องกัน Error CORS)

# ==========================================
# ⚙️ 2. ตั้งค่าระบบ AI และวิดีโอ/กล้อง
# ==========================================
USE_CAMERA = False   # 🔴 เปลี่ยนเป็น True เมื่อต้องการใช้กล้องจริงตอนพรีเซนต์
CAMERA_INDEX = 1     # 0 = กล้องโน้ตบุ๊ก, 1 หรือ 2 = กล้อง USB
VIDEO_PATH = 'test_video.mp4'      # ชื่อไฟล์วิดีโอจำลอง
ZONES_FILE = 'parking_zones.json'  # ชื่อไฟล์บันทึกพิกัดกรอบช่องจอด

# โครงสร้างข้อมูลช่องจอด (แก้ไขให้ทุกช่อง "ว่าง" = True เริ่มต้นทั้งหมด)
buildings_data = [
    {"id": "A", "slots": [True]*10},
    {"id": "B", "slots": [True]*10},
    {"id": "C", "slots": [True]*10},
    {"id": "D", "slots": [True]*10},
    {"id": "E", "slots": [True]*10},
    {"id": "F", "slots": [True]*10},
    {"id": "G", "slots": [True]*10}
]

# ตัวแปรสำหรับระบบวาดกรอบช่องจอดเอง (Dynamic ROI)
parking_zones = {}
edit_mode = False
drawing = False
temp_box = None

# ==========================================
# 🧠 3. ฟังก์ชันระบบบันทึกและวาดช่องจอดด้วยเมาส์
# ==========================================
def load_zones():
    """โหลดพิกัดช่องจอดที่เคยบันทึกไว้ขึ้นมาใช้งานเมื่อเปิดโปรแกรม"""
    global parking_zones
    if os.path.exists(ZONES_FILE):
        with open(ZONES_FILE, 'r') as f:
            data = json.load(f)
            parking_zones = {int(k): tuple(v) for k, v in data.items()}
            print("✅ โหลดพิกัดช่องจอดเรียบร้อยแล้ว")
    else:
        parking_zones = {}

def save_zones():
    """บันทึกพิกัดช่องจอดที่วาดใหม่ลงไฟล์ JSON"""
    with open(ZONES_FILE, 'w') as f:
        json.dump(parking_zones, f)
    print("💾 บันทึกพิกัดช่องจอดลงไฟล์เรียบร้อยแล้ว!")

def check_intersection(car_box, zone_box):
    """เช็คการทับซ้อน โดยดูว่า 'จุดกึ่งกลางของรถ' ตกอยู่ในกรอบช่องจอดหรือไม่"""
    car_center_x = (car_box[0] + car_box[2]) / 2
    car_center_y = (car_box[1] + car_box[3]) / 2
    if zone_box[0] <= car_center_x <= zone_box[2] and zone_box[1] <= car_center_y <= zone_box[3]:
        return True
    return False

def mouse_callback(event, x, y, flags, param):
    """ระบบจัดการเมาส์สำหรับ วาด (คลิกซ้ายลาก) และ ลบ (คลิกขวา)"""
    global drawing, temp_box, parking_zones, edit_mode
    if not edit_mode: return

    if event == cv2.EVENT_LBUTTONDOWN:       # 1. เริ่มคลิกซ้ายค้าง
        drawing = True
        temp_box = [x, y, x, y]
    elif event == cv2.EVENT_MOUSEMOVE:       # 2. ระหว่างลากเมาส์
        if drawing and temp_box is not None:
            temp_box[2], temp_box[3] = x, y
    elif event == cv2.EVENT_LBUTTONUP:       # 3. ปล่อยคลิกซ้าย (สิ้นสุดการวาด)
        drawing = False
        if temp_box is not None:
            x1, y1 = min(temp_box[0], temp_box[2]), min(temp_box[1], temp_box[3])
            x2, y2 = max(temp_box[0], temp_box[2]), max(temp_box[1], temp_box[3])
            if x2 - x1 > 20 and y2 - y1 > 20: # ขนาดต้องใหญ่พอ ป้องกันเผลอคลิก
                new_id = len(parking_zones)
                parking_zones[new_id] = (x1, y1, x2, y2)
            temp_box = None
    elif event == cv2.EVENT_RBUTTONDOWN:     # 4. คลิกขวาเพื่อลบกรอบทิ้ง
        to_delete = None
        for idx, box in parking_zones.items():
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                to_delete = idx
                break
        if to_delete is not None:
            del parking_zones[to_delete]
            reindexed = {i: v for i, v in enumerate(parking_zones.values())} # จัดเรียงลำดับ ID ใหม่
            parking_zones.clear()
            parking_zones.update(reindexed)

# ==========================================
# 📸 4. ฟังก์ชันหลักสำหรับรัน AI ตรวจจับรถยนต์
# ==========================================
def run_yolo():
    global edit_mode
    model = YOLO('yolov8n.pt')  
    load_zones() # โหลดช่องจอดที่บันทึกไว้
    
    if USE_CAMERA:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"🎥 กำลังเปิดกล้อง (Index {CAMERA_INDEX})...")
    else:
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"🎬 กำลังเล่นวิดีโอจำลอง...")
        
    cv2.namedWindow("SpotIQ Camera")
    cv2.setMouseCallback("SpotIQ Camera", mouse_callback)
    
    while True:
        success, frame = cap.read()
        if not success:
            if not USE_CAMERA:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # วนลูปวิดีโอใหม่เมื่อจบ
                continue
            else:
                break
                
        # 🎯 ตรวจจับเฉพาะรถ (ตัดคน/สัตว์ทิ้ง) 
        # แก้ไข conf=0.15 เพื่อให้ AI ตรวจจับรถสีดำหรือสีเทากลืนกับพื้นได้แม่นยำขึ้น
        results = model(frame, classes=[2, 3, 5, 7], conf=0.15) 
        building_a = next(b for b in buildings_data if b["id"] == "A")
        
        num_zones = len(parking_zones)
        slot_occupied = [False] * num_zones if num_zones > 0 else []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_box = (x1, y1, x2, y2)
                
                if not edit_mode:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # วาดกรอบสีฟ้าคลุมตัวรถ
                
                for idx, zone_box in parking_zones.items():
                    if check_intersection(car_box, zone_box):
                        slot_occupied[idx] = True # พบรถในช่องจอด
                        
        # อัปเดตข้อมูลอาคาร A ไปให้หน้าเว็บ (ช่องที่ไม่มีรถ=ว่าง=True)
        building_a["slots"] = [not occ for occ in slot_occupied]

        # วาดกรอบช่องจอดลงบนภาพ
        for idx, zone_box in parking_zones.items():
            is_occupied = slot_occupied[idx] if idx < len(slot_occupied) else False
            color = (0, 0, 255) if is_occupied else (0, 255, 0) # แดง=มีรถ, เขียว=ว่าง
            if edit_mode: color = (0, 255, 255)                 # เหลือง=กำลังแก้ไข
                
            cv2.rectangle(frame, (zone_box[0], zone_box[1]), (zone_box[2], zone_box[3]), color, 2)
            cv2.putText(frame, f"A-{idx+1:02d}", (zone_box[0], zone_box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # วาดกรอบชั่วคราวเวลาลากเมาส์
        if drawing and temp_box is not None:
            cv2.rectangle(frame, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), (255, 0, 255), 2)

        # แสดงเมนูคำแนะนำบนจอ
        if edit_mode:
            cv2.putText(frame, "EDIT MODE: Drag to draw | Right-Click to delete", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'S' to Save | 'C' to Clear | 'E' to Exit Edit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Press 'E' to Edit Slots | 'Q' to Quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("SpotIQ Camera", frame)
        
        # ระบบปุ่มคีย์บอร์ด
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'): break
        elif key == ord('e') or key == ord('E'): edit_mode = not edit_mode
        elif key == ord('s') or key == ord('S'): 
            if edit_mode: save_zones()
        elif key == ord('c') or key == ord('C'): 
            if edit_mode: parking_zones.clear()

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
# 🌐 5. เปิด API ให้เว็บไซต์เข้ามาดึงข้อมูล
# ==========================================
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"buildings": buildings_data})

if __name__ == '__main__':
    # รัน AI แยกส่วนเป็นพื้นหลัง (Threading) ไม่ให้เว็บค้าง
    t = threading.Thread(target=run_yolo)
    t.daemon = True
    t.start()
    
    print("🚀 ระบบ API รันแล้วที่ http://localhost:3000/api/status")
    app.run(host='0.0.0.0', port=3000)
