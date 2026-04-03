from facenet_pytorch import InceptionResnetV1
import torch
import os
#model_path = r"C:\deepfake-hybrid-detector\models\xception-b5690688.pth"
model_path = r"C:\deepfake-hybrid-detector\models\resnetinceptionv1_epoch_32.pth"
#model_path = "C:\deepfake-hybrid-detector\models\deepfake_model.pth"

model = None

def load_model():
    global model
    if model is None:
        checkpoint = torch.load(os.path.join("", model_path), map_location="cpu")

        # Use the same output as checkpoint: 1 class
        model = InceptionResnetV1(pretrained=None, classify=True, num_classes=1)

        # Load weights except logits (last layer) to avoid mismatch
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and "logits" not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        model.eval()

    return model