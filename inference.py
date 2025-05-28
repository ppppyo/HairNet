import torch
from PIL import Image
from torchvision import transforms
from models.cnn_model import CNNModel
from utils.config import config

def load_model(model_path, device):
    model = CNNModel(num_classes=config["num_classes"]).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_class_names = checkpoint.get("class_names", list(range(config["num_classes"])))

    # 클래스 이름 딕셔너리를 index → name 형태로 뒤집기
    if isinstance(raw_class_names, dict):
        class_names = {v: k for k, v in raw_class_names.items()}
    else:
        class_names = raw_class_names

    return model, class_names

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((config["input_size"], config["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # (1, C, H, W)

def inference(image_path, model_path="female_perm.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model, class_names = load_model(model_path, device)
    input_tensor = preprocess_image(image_path).to(device)


    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        prob = probs[0][pred_class_idx].item()

    print(f"✅ 예측 클래스: {class_names[pred_class_idx]} (확률: {prob*100:.2f}%)")

if __name__ == "__main__":
    # 테스트할 이미지 경로 입력
    test_image_path = "test_images/sample2.jfif"  # <-- 여기에 이미지 경로 지정
    inference(test_image_path)
