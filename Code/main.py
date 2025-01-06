import os
import password
import tracker
from picamera2 import Picamera2
import cv2+


if __name__== "__main__":
	picam = Picamera2()
	picam.preview_configuration.main.size=(1280, 720)
	picam.preview_configuration.main.format="RGB888"
	picam.preview_configuration.align()
	picam.configure("preview")
	picam.start()
	password.state_machine(picam)
	cv2.destroyAllWindows()
	tracker.track(picam)
