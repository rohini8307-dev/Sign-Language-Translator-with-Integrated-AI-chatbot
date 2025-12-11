import cv2
import os

# -------- CONFIG --------
labels = ['A','B','C','D','E','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']   # your classes
images_per_label = 200
pause_after = 100
img_size = 300
save_path = "ddatasett"

# -------- CREATE FOLDERS --------
for label in labels:
    os.makedirs(os.path.join(save_path, label), exist_ok=True)

# -------- CAMERA SETUP --------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("\n Camera ready. Show your signs in front of it.")
print(" Controls: 's' → start/resume, 'n' → next label, 'q' → quit\n")

for label in labels:
    count = 0
    print(f"\n Ready to capture for label: '{label}'")
    print("Press 's' to start capturing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Label: {label} | Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):  # Start/resume
            while count < images_per_label:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                resized = cv2.resize(frame, (img_size, img_size))
                cv2.imwrite(os.path.join(save_path, label, f"{count}.jpg"), resized)
                count += 1

                cv2.putText(frame, f"Label: {label} | Count: {count}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Capture", frame)

                if count == pause_after:
                    print(f"Paused at {pause_after}. Press 's' to continue or 'q' to quit.")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

        elif key == ord('n'):
            print(f"✅ Finished capturing for label '{label}' ({count}/{images_per_label})")
            break
        elif key == ord('q'):
            print("Exiting capture.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Dataset capture complete for all labels!")
