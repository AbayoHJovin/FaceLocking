import cv2


def run():
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(haar_file)

    if detector.empty():
        raise RuntimeError(f"Unable to load Haar cascade from {haar_file}")

    camera = cv2.VideoCapture(1)
    if not camera.isOpened():
        raise RuntimeError("Could not access camera. Try index 0, 1, or 2.")

    print("Minimal Haar face detection running. Press 'q' to exit.")

    while True:
        success, image = camera.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detected = detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (fx, fy, fw, fh) in detected:
            cv2.rectangle(
                image,
                (fx, fy),
                (fx + fw, fy + fh),
                (0, 255, 0),
                2,
            )

        cv2.imshow("Haar Face Detection", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
