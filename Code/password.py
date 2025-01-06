import cv2
import numpy as np
from picamera2 import Picamera2
import os

def detect_color(light_color, dark_color, hsv_img):
    
    mask = cv2.inRange(hsv_img,light_color,dark_color)
    segmented = cv2.bitwise_and(hsv_img,hsv_img,mask=mask)
    segmented_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Tamaño del kernel puede ajustarse
    eroded = cv2.erode(segmented_gray, kernel, iterations=6)  # Ajustar las iteraciones según necesidad
    
    contours,_ = cv2.findContours(eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        corners = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True
        )

        if len(corners) == 4:
            return True
        
    return False


def detect_figures(img):
    """
    Detects predefined figures in the given image.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV color ranges for detection
    color_ranges = {
        "yellow": {"light": (19, 85, 96), "dark": (47, 255, 255)},
        "green": {"light": (35, 50, 70), "dark": (85, 255, 255)},
        "blue": {"light": (75, 67, 73), "dark": (145, 255, 177)},
        "brown": {"light": (0, 40, 30), "dark": (28, 231, 100)}
    }
	
    for color, ranges in color_ranges.items():
        if detect_color(ranges["light"], ranges["dark"], hsv_img):
            return color
    return None


def state_machine(picam):
    """
    State machine that processes the sequence of color detection.
    The correct sequence is: Yellow -> Green -> Blue -> Brown.
    """
    state_sequence = ["brown","yellow", "green", "blue"]
    current_state = 0  # Start with the first color (yellow)

    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("CAPTURAMOS")
            # Captura 5 frames consecutivos
            frames = []
            for _ in range(5):
                frame = picam.capture_array()
                frames.append(frame)

            # Verifica si al menos 3 de los 5 frames contienen el color esperado
            correct_count = 0
            for f in frames:
                detected_color = detect_figures(f)
                if detected_color == state_sequence[current_state]:
                    correct_count += 1

            # Si se detectan al menos 3 frames correctos, avanzar al siguiente estado
            if correct_count >= 3:
                print(f"Correct! Detected {state_sequence[current_state]}.")
                current_state += 1
                if current_state == len(state_sequence):
                    print("Password correct! Unlocked.")
                    break  # Se desbloqueó, termina el proceso
            else:
                print(f"Incorrect detection. Going back to the first state.")
                current_state = 0  # Si no se detecta correctamente, reinicia al primer color

        # Espera 1ms para continuar con el siguiente ciclo
        cv2.waitKey(1)

if __name__ == "__main__":
	picam = Picamera2()
	picam.preview_configuration.main.size=(1280, 720)
	picam.preview_configuration.main.format="RGB888"
	picam.preview_configuration.align()
	picam.configure("preview")
	picam.start()
	state_machine(picam)
