import os
import subprocess

def download_data():
    repo_url = "https://github.com/ppppyo/hairNet.git"
    clone_target = "dataset"

    # 1. Clone GitHub repo
    if not os.path.exists(clone_target):
        subprocess.run(["git", "clone", repo_url, clone_target], check=True)
    else:
        print("이미 다운로드된 데이터입니다.")

    # 2. Create flat folder: data/dataset/man_data_flat/
    print("'augmented_1000' 이미지들을 class 폴더로 정리 중...")

    original_dir = os.path.join(clone_target, "dataset", "woman_data")
    flat_dir = "data/woman_data"
    os.makedirs(flat_dir, exist_ok=True)

    if not os.path.exists(original_dir):
        print(f"원본 경로가 존재하지 않습니다: {original_dir}")
        return

    for class_name in os.listdir(original_dir):
        aug_dir = os.path.join(original_dir, class_name, "augmented_1000")
        if not os.path.isdir(aug_dir):
            print(f"건너뜀: {aug_dir} 은 폴더가 아님")
            continue

        # 타겟 클래스 폴더 생성
        target_class_dir = os.path.join(flat_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        # 이미지 링크 연결 또는 복사
        for img_file in os.listdir(aug_dir):
            src = os.path.abspath(os.path.join(aug_dir, img_file))
            dst = os.path.join(target_class_dir, img_file)
            try:
                if not os.path.exists(dst):
                    os.symlink(src, dst)  # Windows에서는 작동안할 수 있음
            except Exception as e:
                print(f"링크 실패: {src} → {dst} / 이유: {e}")

    print(f"완료: {flat_dir} 에 학습용 폴더 구성됨!")

if __name__ == "__main__":
    download_data()
