import os
import sys
import random
from PIL import Image, ImageEnhance

def augment_image(img):
    """무작위로 한 가지 증강 적용 + 최종적으로 256x256 리사이즈"""
    aug_type = random.choice(['flip', 'rotate', 'color', 'brightness', 'contrast'])
    if aug_type == 'flip':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == 'rotate':
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=True)
    elif aug_type == 'color':
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    elif aug_type == 'brightness':
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    elif aug_type == 'contrast':
        enhancer = ImageEnhance.Contrast(img)
        factor = random.uniform(0.8, 1.2)
        img = enhancer.enhance(factor)
    
    # 마지막에 강제 리사이즈
    return img.resize((256, 256))

def get_existing_indices(files, basename):
    """basename_숫자.jpg 형식에서 숫자 모음 리턴 (정수 리스트)"""
    indices = []
    for f in files:
        if f.startswith(basename + "_") and f.endswith(".jpg"):
            suffix = f[len(basename) + 1:-4]
            if suffix.isdigit():
                indices.append(int(suffix))
    return sorted(indices)

def main(root_dir):
    if not os.path.isdir(root_dir):
        print(f"{root_dir} 디렉토리가 존재하지 않습니다.")
        return

    # root_dir 하위에서 이름이 '256'으로 끝나는 디렉토리만 처리
    for subdir in sorted(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        if not subdir.endswith("256"):
            continue

        print(f"처리 중: {subdir_path}")

        # 증강 이미지 저장 폴더
        augmented_dir = os.path.join(subdir_path, "augmented_700")
        if os.path.exists(augmented_dir):
            print(f"{augmented_dir} 이미 존재하므로 건너뜀")
            continue
        os.makedirs(augmented_dir, exist_ok=True)

        # jpg 파일 목록
        files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith('.jpg')])
        if len(files) == 0:
            print("이미지가 없습니다.")
            continue

        # basename 별로 그룹핑 (예: image_1.jpg -> basename = image)
        grouped = {}
        for f in files:
            if "_" in f:
                basename = f.rsplit("_", 1)[0]
                grouped.setdefault(basename, []).append(f)
            else:
                grouped.setdefault(f[:-4], []).append(f)

        for basename, img_files in grouped.items():
            original_count = len(img_files)
            need = 700 - original_count
            if need <= 0:
                print(f"{basename}: 원본 이미지가 700개 이상입니다. 증강하지 않음.")
                continue

            # 원본 이미지 경로 리스트
            original_paths = [os.path.join(subdir_path, f) for f in img_files]

            # 1~700까지 번호 매김용 리스트
            all_images = []

            # 1. 원본 이미지를 augmented_700 폴더에 번호 붙여 복사
            for i, img_path in enumerate(original_paths, start=1):
                img = Image.open(img_path)
                save_name = f"{basename}_{i}.jpg"
                save_path = os.path.join(augmented_dir, save_name)
                img.save(save_path)
                all_images.append(save_path)

            # 2. 증강 이미지 생성 및 저장 (원본 이미지를 순환하며 증강)
            for i in range(need):
                base_img_path = original_paths[i % original_count]
                img = Image.open(base_img_path)
                aug_img = augment_image(img)
                save_name = f"{basename}_{original_count + i + 1}.jpg"
                save_path = os.path.join(augmented_dir, save_name)
                aug_img.save(save_path)
                all_images.append(save_path)

            print(f"{basename}: 총 700개 이미지 생성 완료 (원본 {original_count} + 증강 {need})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python3 augmentation.py <데이터_루트_디렉토리>")
        sys.exit(1)
    root_directory = sys.argv[1]
    main(root_directory)

