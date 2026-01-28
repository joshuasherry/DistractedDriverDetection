import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load MediaPipe face landmarker
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# 3D model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye
    (43.3, 32.7, -26.0),    # Right eye
    (-28.9, -28.9, -24.1),  # Left mouth
    (28.9, -28.9, -24.1)    # Right mouth
], dtype=np.float64)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        lm = result.face_landmarks[0]

        # Draw green landmark points
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
            pitch, yaw, roll = angles

            # Display head pose
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Simple attention logic
            if abs(yaw) > 20 or pitch < -15:
                status = "OFF ROAD"
                color = (0, 0, 255)
            else:
                status = "ON ROAD"
                color = (0, 255, 0)
            cv2.putText(frame, status, (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    cv2.imshow("Head Pose + Landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
