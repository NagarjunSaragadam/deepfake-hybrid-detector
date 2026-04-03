#from modules.spatial_analysis_epoch import load_model
from modules.model_loader import load_model
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import timm

# Preprocessing for CNN
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


def preprocess(face_img):
    img = transform(face_img)
    img = img.unsqueeze(0)  # add batch dimension
    return img


def spatial_score(face_img):    
    """
    Input: face image (numpy array)
    Output: probability score (0 to 1)
    """

    if face_img is None:
        return 0.0

    model =  load_model()
    

    try:
        img = preprocess(face_img)

        with torch.no_grad():
            output = model(img)

            probs = torch.nn.functional.softmax(output, dim=1)

            fake_prob = probs[0][1].item()  # class 1 = fake

        return fake_prob

    except Exception as e:
        print("Spatial analysis error:", e)
        return 0.0
    
def spatial_score_1(face_img):
    if face_img is None:
        return 0.0

    try:
        model = load_model()
        img = transform(face_img).unsqueeze(0)

        with torch.no_grad():
            output = model(img)  # shape [1, 1]
            fake_prob = torch.sigmoid(output[0][0]).item()  # single logit → probability

        return float(fake_prob)

    except Exception as e:
        print("Spatial analysis error:", e)
        return 0.0