import cv2
import numpy as np
from picamera2 import Picamera2

def track(picam):
    # Crear filtro de Kalman
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32) * 0.03

    track_window = None
    crop_hist = None
    measurement = np.zeros((2, 1), np.float32)

    # Detección automática del objeto inicial
    while True:
        frame = picam.capture_array()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))  # Ajustar rango de color si es necesario
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("Buscando objetos", frame)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w * h > 500:  # Filtro para ignorar objetos pequeños
                track_window = (x, y, w, h)
                cx, cy = x + w // 2, y + h // 2
                kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                kf.errorCovPost = np.eye(4, dtype=np.float32)

                crop = frame[y:y + h, x:x + w].copy()
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                crop_hist = cv2.calcHist([hsv_crop], [0], None, [180], [0, 180])
                cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

                print(f"Objeto detectado automáticamente en: {x}, {y}")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

    # Rastreo del objeto
    while True:
        frame = picam.capture_array()

        input_frame = frame.copy()
        hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)

        # Proyección inversa del histograma
        back_proj = cv2.calcBackProject([hsv], [0], crop_hist, [0, 180], 1)

        # Aplicar MeanShift
        ret, track_window = cv2.meanShift(back_proj, track_window, term_crit)
        x, y, w, h = track_window
        cx, cy = x + w // 2, y + h // 2

        # Predicción y corrección de Kalman
        prediction = kf.predict()
        measurement = np.array([[cx], [cy]], dtype=np.float32)
        kf.correct(measurement)

        # Dibujar posiciones
        cv2.rectangle(input_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.circle(input_frame, (int(prediction[0]), int(prediction[1])), 5, (0, 0, 255), -1)
        cv2.circle(input_frame, (cx, cy), 5, (0, 255, 0), -1)

        # Mostrar resultados
        cv2.imshow("Rastreo y Clasificacion", input_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam.stop()  # Detener la cámara cuando se sale

if __name__ == "__main__":
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    track(picam)
