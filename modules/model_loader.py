import torch
import timm

model_path = "C:\deepfake-hybrid-detector\models\deepfake_model.pth"
#model_path = r"C:\deepfake-hybrid-detector\models\xception-b5690688.pth"


def load_model():

    model = timm.create_model("tf_efficientnet_b4", pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    #model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()

    return model