"""
Simple flow:
webcam -> Haar face detection -> MediaPipe FaceMesh (full image)
-> pick 5 landmarks -> visualize

Run:
python -m src.landmarks

Controls:
q : exit
"""

import cv2
import numpy as np
import mediapipe as mp

# FaceMesh landmark indices (5-point layout)
LM_EYE_L = 33
LM_EYE_R = 263
LM_NOSE = 1
LM_MOUTH_L = 61
LM_MOUTH_R = 291


def run():
    # Load Haar cascade
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(haar_file)

    if face_detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {haar_file}")

    # Initialize MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Failed to access camera. Try another index.")

    print("Running Haar + FaceMesh (5 landmarks). Press 'q' to quit.")

    while True:
        success, image = camera.read()
        if not success:
            break

        height, width = image.shape[:2]

        # Haar face detection
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Draw Haar bounding boxes
        for (fx, fy, fw, fh) in detected_faces:
            cv2.rectangle(
                image,
                (fx, fy),
                (fx + fw, fy + fh),
                (0, 255, 0),
                2
            )

        # FaceMesh processing on full frame
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)

        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark
            selected_ids = [
                LM_EYE_L,
                LM_EYE_R,
                LM_NOSE,
                LM_MOUTH_L,
                LM_MOUTH_R
            ]

            keypoints = []
            for idx in selected_ids:
                lm = landmarks[idx]
                keypoints.append([lm.x * width, lm.y * height])

            points = np.array(keypoints, dtype=np.float32)

            # Ensure correct left/right ordering
            if points[0, 0] > points[1, 0]:
                points[[0, 1]] = points[[1, 0]]
            if points[3, 0] > points[4, 0]:
                points[[3, 4]] = points[[4, 3]]

            # Draw landmarks
            for x, y in points.astype(int):
                cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

            cv2.putText(
                image,
                "5 landmarks",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Landmarks (5pt)", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
