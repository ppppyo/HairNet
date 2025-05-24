# hairNet
CNN 기반 헤어스타일 이미지 분류 프로젝트  
(Pytorch + WandB + GitHub 기반 재현 가능한 실험 환경)


## 프로젝트 구조

```
HairNet/
├── data/                # 데이터셋 다운로드 경로 (Git 추적 제외됨)
├── models/              # CNN 모델 정의
├── train/               # 학습 및 평가 스크립트
├── utils/               # 데이터로더 및 설정 관리
├── main.py              # 선택적 실행 스크립트
├── requirements.txt     # 필요 라이브러리
├── README.md            # 프로젝트 설명서
```
## 설치 및 환경 준비
1. Conda 환경 설정
```
conda create -n hairnet python=3.10 -y
conda activate hairnet
pip install -r requirements.txt
```
2. Jupyter Notebook 실행용 설정(optional)
```
conda install notebook ipykernel -y
python -m ipykernel install --user --name hairnet --display-name "HairNet"
``` 
## 데이터 다운로드
```
python data/download_dataset.py
```
## 모델 학습
```
python -m train.train
```
기능:
- .pth 모델 저장 (best_model.pth, last_model.pth)
- 자동 검증 Accuracy 출력
- config.py로 하이퍼파라미터 설정
- 모델 체크포인트 저장 가능 (checkpoint.pth)

### 구성 및 설정 변경
utils/config.py에서 제어 가능
```
config = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 30,
    "num_classes": 17,
    "data_path": "data/hairNet/dataset/man_data",
    "use_wandb": True,
    "resume": False
}
```
## WandB 실험 추적(optional)
1. https://wandb.ai 가입 및 로그인
2. CLI에서 로그인
```
wandb login
```
3. 학습 시작:
```
python -m train.train_wandb
```
## 모델 평가
```
python evaluate.py
```


