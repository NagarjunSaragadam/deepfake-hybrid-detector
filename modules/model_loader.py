import torch
import timm
import os
from huggingface_hub import login, whoami

def load_model(api_key="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"):
    if api_key:
        login(token=api_key)
        try:
            user_info = whoami(token=api_key, cache=True)
        except Exception:
            pass # Fallback if even the whoami call fails
    model = timm.create_model("tf_efficientnet_b4", pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    model.eval()
    return model