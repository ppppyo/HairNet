import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.cnn_model import CNNModel
from utils.dataset_loader import get_dataloaders
from utils.config import config as base_config

import argparse
import yaml

from tqdm import tqdm

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")  # True만 받아도 되면 이렇게
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pth")
    args = parser.parse_args()

    # config 덮어쓰기
    config = base_config.copy()
    config["resume"] = args.resume
    config["checkpoint_path"] = args.checkpoint_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    early_stop_counter = 0
    patience = 5

    # ✅ 1. wandb 초기화
    if config.get("use_wandb", False):
        print("wandb about to initialize")
        wandb.init(project="hairnet", config=config, reinit=True)
        wandb_config = wandb.config
        print("✅ wandb initialized!")
    else:
        print("❌ wandb disabled!")
        wandb_config = config  # wandb 없이도 동일하게 사용

    # 2. 데이터 로딩
    train_loader, val_loader, test_loader, class_map = get_dataloaders(
        wandb_config["data_path"],
        batch_size=wandb_config["batch_size"]
    )

    print("📋 class_map (index to label):")
    for label, idx in class_map.items():
        print(f"  {idx}: {label}")

    model = CNNModel(num_classes=wandb_config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=wandb_config["learning_rate"],
        weight_decay=wandb_config.get("weight_decay", 1e-4)
    )
    # # hyparams 실험 시
    # optimizer = {
    # "adam": torch.optim.Adam,
    # "adamw": torch.optim.AdamW,
    # "sgd": torch.optim.SGD
    # }[wandb_config.get("optimizer", "adam")](model.parameters(), lr=wandb_config["learning_rate"], weight_decay=wandb_config.get("weight_decay", 1e-4))


    scheduler = None
    if wandb_config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=wandb_config.get("scheduler_T_max", 50),  # 한 주기 길이 (에폭 기준)
            eta_min=wandb_config.get("scheduler_eta_min", 1e-6)  # 최저 학습률
        )
    # # hyparams 실험 시
    # if wandb_config.get("use_scheduler", False):
    #     scheduler = optim.lr_scheduler.StepLR(
    #         optimizer,
    #         step_size=wandb_config.get("scheduler_step", 10),
    #         gamma=wandb_config.get("scheduler_gamma", 0.5)
    #     )

    start_epoch = 0
    best_val_acc = 0.0

    if config.get("resume", False):
        checkpoint = torch.load(config.get("checkpoint_path", "checkpoint.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint.get("best_val_acc", 0.0)
        print(f"🔁 이어서 학습 시작: epoch {start_epoch}부터")

    for epoch in range(start_epoch, wandb_config["epochs"]):
        model.train()
        total_loss = 0.0

        num_batches = 0
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{wandb_config['epochs']}", dynamic_ncols=True)

        for batch_idx, (images, labels) in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            num_batches += 1

            train_bar.set_postfix(loss=f"{loss.item():.4f}", batch=batch_idx)


        # ✅ validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total

        avg_loss = total_loss / num_batches

        # # train 정확도 계산
        # model.eval()
        # train_correct, train_total = 0, 0
        # with torch.no_grad():
        #     for images, labels in train_loader:
        #         images, labels = images.to(device), labels.to(device)
        #         outputs = model(images)
        #         _, predicted = torch.max(outputs, 1)
        #         train_total += labels.size(0)
        #         train_correct += (predicted == labels).sum().item()

        # train_acc = 100 * train_correct / train_total
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | Validation Accuracy: {val_acc:.2f}%")

        # ✅ wandb 로깅
        if config.get("use_wandb", False):
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_accuracy": val_acc
            })

        # ✅ 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0  
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_map,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc
            }, f"best_model_woman_all.pth")
            print("✅ Best model saved!")

        # # hyparams 실험 시
        # if config.get("use_wandb", False):
        #     wandb.save("best_model.pth")
        # else:
        #     early_stop_counter += 1 

        # # early stopping 조건 검사
        # if early_stop_counter >= patience:
        #     print("🛑 Early stopping triggered!")
        #     break

        # checkpoint = {
        #     "epoch": epoch,
        #     "model_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        #     "best_val_acc": best_val_acc
        # }

        # torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pth")

        if epoch + 1 == wandb_config["epochs"]:
            torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }, f"last_model_woman_all_0528_{epoch+1}.pth")

        if scheduler:
            scheduler.step()

    # # test set 성능 평가
    # model.eval()
    # test_correct, test_total = 0, 0
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, 1)
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()

    # test_acc = 100 * test_correct / test_total
    # print(f" Test Accuracy: {test_acc:.2f}%")

    # # wandb 로깅도 원하면 추가
    # if config.get("use_wandb", False):
    #     wandb.log({"test_accuracy": test_acc})


if __name__ == "__main__":
    train()

    # # hyparams 실험 시
    # if config.get("use_wandb", False):
    #     wandb.finish()
