import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import pyttsx3
import ollama
import textwrap
from threading import Thread

model = load_model("signn_language_landmark_model.h5")
class_names = [chr(i) for i in range(65, 91)]  # 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
predicted_text = ""
prev_letter = ""
last_detect_time = time.time()
cooldown = 2.0
no_hand_start = None
space_delay = 2.0
ai_reply = ""

print("Sign to Sentence Chat Active!")
print("Hold each letter steady for 2s.")
print("Remove hand for 2s to add SPACE.")
print("Press BACKSPACE to delete, ENTER to send, Q to quit.\n")

def draw_wrapped_text(img, text, pos, color, scale, thickness, max_width):
    y = pos[1]
    for line in textwrap.wrap(text, width=max_width):
        cv2.putText(img, line, (pos[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)
        y += int(30 * scale)

engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak(text):
    def _talk():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("Speech error:", e)

    Thread(target=_talk, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
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
                prediction = model.predict(np.array([row]), verbose=0)
                pred_idx = np.argmax(prediction)
                letter = class_names[pred_idx]

                # Stable detection logic
                if letter == prev_letter and (current_time - last_detect_time > cooldown):
                    predicted_text += letter
                    last_detect_time = current_time
                    prev_letter = ""
                    print(f"Added: {letter}")
                elif letter != prev_letter:
                    prev_letter = letter
                    last_detect_time = current_time

                cv2.putText(frame, f"{letter}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        if no_hand_start is None:
            no_hand_start = time.time()
        elif time.time() - no_hand_start > space_delay:
            if len(predicted_text) > 0 and predicted_text[-1] != " ":
                predicted_text += " "
                print("Added: space")
            no_hand_start = None

    draw_wrapped_text(frame, f"You: {predicted_text}", (10, 400),
                      (255, 255, 255), 0.8, 2, 50)
    draw_wrapped_text(frame, f"Chatbot: {ai_reply}", (10, 460),
                      (0, 255, 255), 0.8, 2, 50)

    cv2.imshow("Sign Language Chat", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break

    # Backspace
    elif key == 8:
        if predicted_text:
            predicted_text = predicted_text[:-1]
            print("Deleted last character")

    elif key == 13:
        if predicted_text.strip():
            print(f"\nYou: {predicted_text}")

            try:
                response = ollama.chat(model="mistral:latest", messages=[
                    {
                        "role": "system",
                        "content": "You are a friendly assistant. Respond in short, natural English."
                    },
                    {
                        "role": "user",
                        "content": predicted_text
                    }
                ])

                ai_reply = response["message"]["content"].strip()
                print(f"Chatbot: {ai_reply}\n")

                speak(ai_reply)

            except Exception as e:
                ai_reply = "Error!"
                print(e)

            predicted_text = "" 

cap.release()
cv2.destroyAllWindows()
print("Chat ended.")

