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
        "brown": {"light": (0, 34, 63), "dark": (25, 200, 131)}
    }
	
    for color, ranges in color_ranges.items():
        if detect_color(ranges["light"], ranges["dark"], hsv_img):
            return color
    return None
    
if __name__ =="__main__":
    filenames = [os.path.join("../",f"Imagen_capturada_{i}.jpg") for i in range(11,15)]
    imgs = [cv2.imread(filename) for filename in filenames]
    colores = []
    for img in imgs:
	    colores.append(detect_figures(img))
    print(colores)
