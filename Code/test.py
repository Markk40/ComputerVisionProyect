import cv2
import os
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            path = "../"
            os.makedirs(path, exist_ok=True)
            i = len(os.listdir(path))
            img_path = os.path.join(path, f"Imagen_capturada_{i}.jpg")
            
            cv2.imwrite(img_path,frame)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
