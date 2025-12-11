from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import time
import ollama

app = Flask(__name__)

model = load_model("signn_language_landmark_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # A-Z

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

predicted_text = ""
prev_letter = ""
lock = threading.Lock()

camera = None
last_detect_time = time.time()
cooldown = 1.5
no_hand_start = None
space_delay = 2.0
frame_skip = 2  

def generate_frames():
    global predicted_text, prev_letter, last_detect_time, no_hand_start, camera

    if camera is None:
        camera = cv2.VideoCapture(0)

    frame_count = 0
    last_letter = ""
    last_letter_time = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            current_time = time.time()

            if results.multi_hand_landmarks:
                no_hand_start = None
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    x_vals = [lm.x for lm in hand_landmarks.landmark]
                    y_vals = [lm.y for lm in hand_landmarks.landmark]
                    min_x, max_x = min(x_vals), max(x_vals)
                    min_y, max_y = min(y_vals), max(y_vals)

                    row = []
                    for lm in hand_landmarks.landmark:
                        norm_x = (lm.x - min_x) / (max_x - min_x)
                        norm_y = (lm.y - min_y) / (max_y - min_y)
                        row += [norm_x, norm_y]

                    if len(row) == 42:
                        pred = model.predict(np.array([row]), verbose=0)
                        letter = class_names[np.argmax(pred)]
                        last_letter = letter
                        last_letter_time = current_time
                        if letter == prev_letter and (current_time - last_detect_time > cooldown):
                            with lock:
                                predicted_text += letter
                            last_detect_time = current_time
                            prev_letter = ""
                        elif letter != prev_letter:
                            prev_letter = letter
                            last_detect_time = current_time

            else:
        
                if no_hand_start is None:
                    no_hand_start = time.time()
                elif time.time() - no_hand_start > space_delay:
                    with lock:
                        if predicted_text and predicted_text[-1] != " ":
                            predicted_text += " "
                    no_hand_start = None


        with lock:
            display_text = predicted_text

        if time.time() - last_letter_time < 1.2:
            cv2.putText(frame, f"Letter: {last_letter}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    with lock:
        return jsonify({"text": predicted_text})

@app.route('/clear_text')
def clear_text():
    global predicted_text
    with lock:
        predicted_text = ""
    return jsonify({"text": ""})

@app.route('/send_text', methods=['POST'])
def send_text():
    global predicted_text
    data = request.get_json()
    user_text = data.get("text", "")

    if not user_text.strip():
        return jsonify({"reply": "No message entered."})

    try:
        response = ollama.chat(model="mistral:latest", messages=[
            {"role": "system", "content": "You are a friendly personal assistant. Respond in short natural English and with consice responses of 2 sentences."},
            {"role": "user", "content": user_text}
        ])
        ai_reply = response["message"]["content"].strip()


        def speak(msg):
            engine = pyttsx3.init()
            engine.say(msg)
            engine.runAndWait()

        threading.Thread(target=speak, args=(ai_reply,), daemon=True).start()
        with lock:
            predicted_text = ""

        return jsonify({"reply": ai_reply})

    except Exception as e:
        print("⚠️ Ollama Error:", e)
        return jsonify({"reply": "⚠️ AI model unavailable."})

@app.route('/shutdown_camera')
def shutdown_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "Camera released"})
if __name__ == "__main__":
    app.run(debug=False, threaded=True)
