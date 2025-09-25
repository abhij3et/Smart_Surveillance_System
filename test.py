import time, threading, cv2, numpy as np, tensorflow as tf
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify
import telegram, asyncio
from datetime import datetime
from pymongo import MongoClient

# --- Config ---
ALERT_COOLDOWN = 12
DETECTION_CONF_THRESHOLD = 0.20
status_lock = threading.Lock()
last_weapon_detection_time = None
last_weapon_info = "Safe"
last_violence_detection_time = None
last_violence_info = "Safe"
crowd_count = "Calculating..."
crowd_history = []

# Telegram Config
TELEGRAM_BOT_TOKEN = "8465770268:AAHspvpjMrQJXA1Bmg0zGIISrseKhJrdcUw"
TELEGRAM_CHAT_ID = "6594618388"
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# MongoDB Config
mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["SecurityAlerts"]
collection = db["Detections"]

app = Flask(__name__)

# Load models
yolo_crowd_model = YOLO('CrowdDetection/best.pt')
yolo_weapon_model = YOLO('Weapon_Detection/weapon.pt')
violence_model = tf.keras.models.load_model('CriminalAct_Detection/models/violence_detection_model.h5')

cap = cv2.VideoCapture(0)
frame_lock = threading.Lock()
latest_frame = None
processed_frames = {'crowd': None, 'weapon': None, 'violence': None}

# ================= TELEGRAM ALERT HANDLER =================
async def send_telegram_async(msg, frame=None):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        if frame is not None:
            _, buffer = cv2.imencode(".jpg", frame)
            await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=buffer.tobytes())
    except Exception as e:
        print("Telegram error:", e)

def send_telegram_alert(msg, frame=None):
    asyncio.run(send_telegram_async(msg, frame))

# ================= MONGODB ALERT SAVER =================
def save_alert_to_db(alert_type, location="Camera 1", confidence=None, people_count=None):
    now = datetime.now()
    document = {
        "type": alert_type,
        "confidence": confidence,
        "people_count": people_count,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "location": location
    }
    collection.insert_one(document)

# ================= THREADS =================
def read_frames():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret: break
        with frame_lock:
            latest_frame = frame.copy()
        time.sleep(0.01)

def crowd_detection():
    global crowd_count, crowd_history
    last_crowd_alert_time = None
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        annotated = frame.copy()
        results = yolo_crowd_model.track(annotated, conf=0.5, persist=True, tracker="bytetrack.yaml")
        people_count = 0
        if results and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().tolist()
            people_count = len(set(track_ids))
        crowd_history.append(people_count)
        if len(crowd_history) > 30:
            crowd_history.pop(0)
        if people_count > 35:
            crowd_count = f"ALERT: Too many people! ({people_count})"
            now = time.time()
            if not last_crowd_alert_time or (now - last_crowd_alert_time >= ALERT_COOLDOWN):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                send_telegram_alert(f"🚨 CROWD ALERT at {timestamp}\nPeople Count: {people_count}", annotated)
                save_alert_to_db(alert_type="Crowd", people_count=people_count)
                last_crowd_alert_time = now
        else:
            crowd_count = f"{min(crowd_history)}-{max(crowd_history)}"
        annotated = results[0].plot() if results else annotated
        cv2.putText(annotated, f'Current Count: {people_count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        with frame_lock:
            processed_frames['crowd'] = annotated
        time.sleep(0.01)

def weapon_detection():
    global last_weapon_detection_time, last_weapon_info
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        annotated = frame.copy()
        results = yolo_weapon_model(annotated, conf=0.75)
        if results:
            res = results[0]
            annotated = res.plot()
            weapon_detected = False
            detected_info = None
            for box in res.boxes:
                conf = float(box.conf[0].item()) if hasattr(box.conf,"__len__") else float(box.conf)
                cls = int(box.cls[0].item()) if hasattr(box.cls,"__len__") else int(box.cls)
                name = yolo_weapon_model.names.get(cls,str(cls))
                if name.lower() in ['gun','knife','handgun'] and conf>=DETECTION_CONF_THRESHOLD:
                    weapon_detected = True
                    detected_info = f"UNSAFE: {name} ({conf:.2f})"
                    break
            if weapon_detected:
                now = time.time()
                if not last_weapon_detection_time or (now - last_weapon_detection_time >= ALERT_COOLDOWN):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    send_telegram_alert(f"🚨 WEAPON DETECTED at {timestamp}\n{detected_info}", annotated)
                    save_alert_to_db(alert_type="Weapon", confidence=conf)
                    with status_lock:
                        last_weapon_detection_time = now
                        last_weapon_info = detected_info
        with frame_lock:
            processed_frames['weapon'] = annotated
        time.sleep(0.01)

def violence_detection():
    global last_violence_detection_time, last_violence_info
    IMG_H, IMG_W = 150, 150
    frame_rate = 15
    count = 0
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        count += 1
        annotated = frame.copy()
        if count % frame_rate == 0:
            img = cv2.resize(annotated,(IMG_W,IMG_H))
            img = np.expand_dims(img, axis=0)/255.0
            pred = violence_model.predict(img, verbose=0)[0][0]
            is_fight = pred > 0.5
            conf = pred if is_fight else (1-pred)
            if is_fight:
                now = time.time()
                if not last_violence_detection_time or (now - last_violence_detection_time >= ALERT_COOLDOWN):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    send_telegram_alert(f"🚨 FIGHT DETECTED at {timestamp}\nConfidence: {conf:.2f}", annotated)
                    save_alert_to_db(alert_type="Fight", confidence=conf)
                    with status_lock:
                        last_violence_detection_time = now
                        last_violence_info = f"ALERT: FIGHT ({conf:.2f})"
            cv2.putText(annotated, f"Prediction:{'fight' if is_fight else 'not_fight'}({conf:.2f})",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            if is_fight:
                cv2.putText(annotated,"ALERT: FIGHT DETECTED",(10,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        with frame_lock:
            processed_frames['violence'] = annotated
        time.sleep(0.01)

# ================= FLASK STREAMS =================
def generate_frames(stream_type):
    while True:
        with frame_lock:
            frame = processed_frames.get(stream_type)
        if frame is not None:
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        else:
            time.sleep(0.03)

@app.route('/crowd_feed')
def crowd_feed(): return Response(generate_frames('crowd'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/weapon_feed')
def weapon_feed(): return Response(generate_frames('weapon'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/violence_feed')
def violence_feed(): return Response(generate_frames('violence'), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/get_status')
def get_status():
    now = time.time()
    with status_lock:
        weapon_status = last_weapon_info if last_weapon_detection_time and now-last_weapon_detection_time <= ALERT_COOLDOWN else "Safe"
        violence_status = last_violence_info if last_violence_detection_time and now-last_violence_detection_time <= ALERT_COOLDOWN else "Safe"
    return jsonify({'crowd_count': crowd_count, 'weapon_status': weapon_status, 'violence_status': violence_status})

# ================= FLASK PAGES =================
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analytics.html')
def analytics():
    return render_template('analytics.html')

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    threading.Thread(target=read_frames, daemon=True).start()
    threading.Thread(target=crowd_detection, daemon=True).start()
    threading.Thread(target=weapon_detection, daemon=True).start()
    threading.Thread(target=violence_detection, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)

    