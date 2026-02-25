import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ------------------- Initialize MediaPipe Face Landmarker -------------------
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

# ------------------- Head pose 3D model points -------------------
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye
    (43.3, 32.7, -26.0),    # Right eye
    (-28.9, -28.9, -24.1),  # Left mouth
    (28.9, -28.9, -24.1)    # Right mouth
], dtype=np.float64)

cap = cv2.VideoCapture(0)

# ------------------- State variables -------------------
baseline_collected = False
baseline_yaw = 0
baseline_pitch = 0
baseline_roll = 0
calibration_data = []
calibration_duration = 5  # seconds

# ------------------- Main Loop -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    yaw = pitch = roll = 0  # defaults if detection fails

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Draw landmarks
        for point in lm:
            x = int(point.x * w)
            y = int(point.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Head pose points
        image_points = np.array([
            (lm[1].x * w, lm[1].y * h),     # Nose
            (lm[152].x * w, lm[152].y * h), # Chin
            (lm[33].x * w, lm[33].y * h),   # Left eye
            (lm[263].x * w, lm[263].y * h), # Right eye
            (lm[61].x * w, lm[61].y * h),   # Left mouth
            (lm[291].x * w, lm[291].y * h)  # Right mouth
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if success:
            rot_mat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
            pitch, yaw, roll = angles  # degrees

    # ------------------- Baseline calibration -------------------
    if not baseline_collected:
        cv2.putText(frame, "Press SPACE to start baseline calibration", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == 32 and not baseline_collected:  # SPACE
        calibration_data = []
        start_time = time.time()
        print("Calibration started for 5 seconds...")
        while time.time() - start_time < calibration_duration:
            ret2, frame2 = cap.read()
            if not ret2:
                continue
            h2, w2, _ = frame2.shape
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
            result2 = landmarker.detect(mp_image2)

            # Calculate remaining time for countdown
            remaining = int(calibration_duration - (time.time() - start_time) + 1)
            cv2.putText(frame2, f"Calibrating... {remaining}s", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if result2.face_landmarks:
                lm2 = result2.face_landmarks[0]
                image_points2 = np.array([
                    (lm2[1].x * w2, lm2[1].y * h2),     # Nose
                    (lm2[152].x * w2, lm2[152].y * h2), # Chin
                    (lm2[33].x * w2, lm2[33].y * h2),   # Left eye
                    (lm2[263].x * w2, lm2[263].y * h2), # Right eye
                    (lm2[61].x * w2, lm2[61].y * h2),   # Left mouth
                    (lm2[291].x * w2, lm2[291].y * h2)  # Right mouth
                ], dtype=np.float64)
                success2, rvec2, tvec2 = cv2.solvePnP(
                    model_points, image_points2, camera_matrix, dist_coeffs
                )
                if success2:
                    rot_mat2, _ = cv2.Rodrigues(rvec2)
                    angles2, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat2)
                    calibration_data.append(angles2)

            cv2.imshow("Driver Monitor", frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Compute baseline
        calibration_data = np.array(calibration_data)
        baseline_pitch = np.median(calibration_data[:, 0])
        baseline_yaw = np.median(calibration_data[:, 1])
        baseline_roll = np.median(calibration_data[:, 2])
        baseline_collected = True
        print(f"Baseline collected! Pitch={baseline_pitch:.1f}, Yaw={baseline_yaw:.1f}, Roll={baseline_roll:.1f}")

    # ------------------- Post-calibration detection -------------------
    if baseline_collected:
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        yaw_offset = yaw - baseline_yaw
        pitch_offset = pitch - baseline_pitch

        if abs(yaw_offset) > 20 or pitch_offset < -15:
            status = "OFF ROAD"
            color = (0, 0, 255)
        else:
            status = "ON ROAD"
            color = (0, 255, 0)
        cv2.putText(frame, status, (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Driver Monitor", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
