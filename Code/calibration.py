from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import matplotlib.pyplot as plt
import os

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def show_image(img,nombre,show=False):
    if show:
        cv2.imshow(nombre,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
def write_image(img, nombre):
    os.makedirs("./imagenes",exist_ok=True)
    cv2.imwrite(f"./imagenes/{nombre}.jpg", img)

def get_chessboard_points(chessboard_shape, dx, dy):
    lista=[]
    for y in range(0,chessboard_shape[1]):
        for x in range(0,chessboard_shape[0]):
            lista.append((x*dx,y*dy,0))
    return np.asarray(lista, dtype=np.float32)


if __name__ == "__main__":
    archivos = os.listdir("./data")
    imgs_path = ["./data/"+archivo for archivo in archivos]
    imgs = load_images(imgs_path)
    print(f"Se cargaron {len(imgs)} imagenes")

    corners = [cv2.findChessboardCorners(img, patternSize=(7,7)) for img in imgs]

    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    corners_refined = [cv2.cornerSubPix(i, cor[1], (7, 7), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy = copy.deepcopy(imgs)

    for i in range(len(imgs_copy)):
        cv2.drawChessboardCorners(imgs_copy[i],patternSize=(7,7), corners=corners[i][1], patternWasFound = corners[i][0])

    for i in range(len(imgs_copy)):
        nombre = f"Image_{i}"
        # show_image(imgs_copy[i],nombre)
        write_image(imgs_copy[i],nombre)
    
    chessboard_points = get_chessboard_points((7, 7), 30, 30)
    object_points = np.asarray([chessboard_points for _ in range(len(corners))],dtype=np.float32)

    
    valid_corners = [cor[1] for cor in corners if cor[0]]
    object_points = np.asarray([chessboard_points for _ in range(len(valid_corners))],dtype=np.float32)
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points,valid_corners,imgs[0].shape[0:2],None,None)
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)

    