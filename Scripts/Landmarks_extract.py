import cv2
import mediapipe as mp
import os
import csv

dataset_dir = r"c:\Users\HAI\ddatasett"
output_file = "landmarkss.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

header = ["label"]
for i in range(21):
    header += [f"x{i}", f"y{i}"]
    
with open(output_file, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(header)

    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    row = [label]
                    for lm in hand_landmarks.landmark:
                        row += [lm.x, lm.y]
                    csv_writer.writerow(row)

print("Landmarks extracted and saved to 'landmarkss.csv'")


