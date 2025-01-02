import os
import cv2
import imageio
import numpy as np
from typing import List
from utils import non_max_suppression, get_hsv_color_ranges
import os

def show_image(img: np.array, img_name: str = "Image"):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    
def pattern(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Define red color
    light_red = (0, 60, 90)
    dark_red = (0, 90, 40)
    
    light_orange = (1, 190, 200)
    dark_orange = (255, 255, 255)

    light_green =(44,0,0)
    dark_green = (98,255,255)
    
    light_brown = (0, 34, 63)
    dark_brown = (25, 200, 131)
    return detect_color(light_brown,dark_brown,hsv_img)


if __name__ =="__main__":
    filename = os.path.join("../","Imagen_capturada_9.jpg")
    img = cv2.imread(filename)
    print(pattern (img))
