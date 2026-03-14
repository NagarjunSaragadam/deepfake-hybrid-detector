import numpy as np
import cv2

def frequency_analysis(face_img):

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = 20 * np.log(np.abs(fshift))

    score = magnitude.mean()

    return score