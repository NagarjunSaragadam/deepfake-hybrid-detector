import torch
import cv2
import numpy as np
from modules.model_loader import load_model

model = load_model()

def predict_deepfake(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))

    img = img / 255.0
    img = np.transpose(img, (2,0,1))

    img = torch.tensor(img).float().unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1)

    return "Fake" if pred.item() == 1 else "Real"