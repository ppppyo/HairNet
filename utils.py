import torch
from PIL import Image
from torchvision import transforms
from models.cnn_model import CNNModel
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib
import matplotlib.pyplot as plt

def load_model(model_path, device, num_classes):
    model = CNNModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_class_names = checkpoint.get("class_names", list(range(num_classes)))

    # 클래스 이름 딕셔너리를 index → name 형태로 뒤집기
    if isinstance(raw_class_names, dict):
        class_names = {v: k for k, v in raw_class_names.items()}
    else:
        class_names = raw_class_names

    return model, class_names

mtcnn=MTCNN(keep_all=True)

def preprocess_image(img, target_size=(256, 256), padding_ratio=1.0):
    """
    파일 경로에 있는 이미지를 로드하고 얼굴이 감지되면 크롭 및 리사이즈, 그렇지 않으면 비율 유지 축소 후 중앙 배치
    리사이즈된 Tensor 객체를 반환합니다.
    """
    w, h = img.size
    boxes, probs = mtcnn.detect(img)

    if boxes is None:
        # 얼굴 없으면 비율 유지하며 축소(뒷모습 등)
        img.thumbnail(target_size, Image.BILINEAR)
        new_img = Image.new('RGB', target_size, (0, 0, 0))
        new_w, new_h = img.size
        left = (target_size[0] - new_w) // 2
        top = (target_size[1] - new_h) // 2
        new_img.paste(img, (left, top))
        return transform(new_img).unsqueeze(0)

    if len(boxes) != 1:
        return None # 얼굴이 여러 개면 None 반환(Error handling)

    (x1, y1, x2, y2) = boxes[0]
    face_w = x2 - x1
    face_h = y2 - y1

    padding_w = face_w * padding_ratio
    padding_h = face_h * padding_ratio

    left = max(0, int(x1 - padding_w))
    top = max(0, int(y1 - padding_h))
    right = min(w, int(x2 + padding_w))
    bottom = min(h, int(y2 + padding_h))

    img_cropped = img.crop((left, top, right, bottom))

    return transform(img_cropped).unsqueeze(0) # PIL -> Tensor 변환

matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),

])

def inference(input_tensor, model_path, num_classes, CLASSES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    model, class_names = load_model(model_path, device, num_classes)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(probs, dim=1).item()
        prob = probs[0][pred_class_idx].item()
        result=round(prob*100,2)
        pred_class=CLASSES[pred_class_idx]
    return pred_class, result

if __name__ == "__main__":
    # 테스트할 이미지 경로 입력
    test_image_path = "test_images/sample2.jfif"  # <-- 여기에 이미지 경로 지정
    inference(test_image_path)
