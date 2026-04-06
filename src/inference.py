import torch
import numpy as np
from model import CNN1D

def predict(features, model_path, num_classes=10):
    model = CNN1D(num_classes=num_classes, input_size=len(features))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(x)
        pred = torch.argmax(output, dim=1)

    return pred.item()


if __name__ == "__main__":
    # Example input (dummy)
    sample = np.random.rand(58)
    result = predict(sample, '../models/model.pth')
    print("Predicted Class:", result)