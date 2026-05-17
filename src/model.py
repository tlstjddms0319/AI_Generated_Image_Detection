"""
model.py
========
ResNet-50 모델 정의 공통 모듈
모든 노트북에서 아래처럼 import해서 사용:

    from src.model import build_resnet50
    model = build_resnet50().to(device)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """
    ImageNet pretrained ResNet-50의 FC 레이어만 교체

    Args:
        num_classes: 분류 클래스 수 (Real/AI = 2)
        pretrained : ImageNet pretrained weight 사용 여부

    Returns:
        model: FC 레이어가 교체된 ResNet-50
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model   = models.resnet50(weights=weights)

    # 마지막 FC 레이어 교체: 2048 → Dropout → num_classes
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def load_model(ckpt_path: str, device: torch.device, num_classes: int = 2) -> nn.Module:
    """
    저장된 체크포인트에서 모델 로드

    Args:
        ckpt_path  : .pth 파일 경로
        device     : 실행 디바이스 (cuda / cpu)
        num_classes: 분류 클래스 수

    Returns:
        model: 가중치가 로드된 ResNet-50 (eval 모드)
    """
    model = build_resnet50(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[Model] 체크포인트 로드 완료: {ckpt_path}")
    return model
