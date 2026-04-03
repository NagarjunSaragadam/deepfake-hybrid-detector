import numpy as np
import cv2

def frequency_analysis(face_img):

     gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

     f = np.fft.fft2(gray)
     fshift = np.fft.fftshift(f)
     magnitude = np.abs(fshift)

     h, w = magnitude.shape
     center_h, center_w = h // 2, w // 2
     radius = min(center_h, center_w) // 4 #30

     y, x = np.ogrid[:h, :w]
     mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2     

     low_freq_magnitude = magnitude[mask].mean()
     high_freq_magnitude = magnitude[~mask].mean()
     score = high_freq_magnitude / (low_freq_magnitude + 1e-8)    

     # Normalize to 0-1 (assume typical FFT mean ranges 0-150)
     score_norm = np.clip(score / 5, 0, 1)

     return score_norm

def frequency_analysis1(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    # Normalize spectrum
    magnitude = magnitude / (np.max(magnitude) + 1e-8)

    # Focus on high-frequency region (important for deepfakes)
    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    high_freq = magnitude[0:center_h//2, 0:center_w//2]

    score = np.mean(high_freq)

    return float(score)