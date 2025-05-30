import torch
from models.cnn_model import CNNModel
from utils.dataset_loader import get_dataloaders
from utils.config import config

def evaluate(model_path="best_model.pth"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # 모델 로드
    model = CNNModel(num_classes=config["num_classes"]).to(device)
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])   
    model.eval()

    # 데이터 로드
    _, _, test_loader, class_map = get_dataloaders(config["data_path"], config["batch_size"])

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    evaluate()
