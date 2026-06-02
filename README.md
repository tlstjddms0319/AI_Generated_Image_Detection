# 🔍 AI-Generated Image Detection
### Explainability & Adversarial Robustness Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

**Team 4** | Deep Learning Project | 2026
김지민 · 박우석 · 성주인 · 신성은

</div>

---

## 📌 Overview

AI 생성 이미지 탐지(DeepFake Detection), 설명 가능한 AI(XAI), 적대적 견고성(Adversarial Robustness) 평가를 하나의 파이프라인으로 통합한 연구입니다.

| Phase | 내용 | 핵심 기술 |
|:---:|:---|:---|
| **Phase 1** | ResNet-50 이진 분류기 학습 | Transfer Learning, AdamW, CosineAnnealing |
| **Phase 2** | Grad-CAM XAI 시각화 | Gradient-weighted Class Activation Mapping |
| **Phase 3** | FGSM 적대적 공격 및 견고성 평가 | Fast Gradient Sign Method, Robustness Index |

---

## 📊 Results

<div align="center">

| 지표 | 값 |
|:---:|:---:|
| **Clean Test AUC** | **0.9546** |
| **Clean Test Accuracy** | **0.8667** |
| FGSM eps=0.01 AUC | 0.0029 |
| FGSM eps=0.01 ASR | **0.9651** |

> AUC 단독 지표로는 adversarial robustness를 과대평가할 수 있습니다.
> ASR(최대 0.9651)과 Accuracy Drop(최대 0.8342)이 실제 취약성을 더 직관적으로 드러냅니다.

</div>

---

## 🗂️ Project Structure

```
deepfake-detection/
├── final_report.ipynb      # 최종 보고서 (전체 파이프라인)
├── notebooks/
│   ├── 01_dataset.ipynb    # 데이터셋 탐색
│   ├── 02_deduplication.ipynb  # 중복 제거
│   ├── 03_train.ipynb      # 모델 학습
│   ├── 04_gradcam.ipynb    # Grad-CAM 시각화
│   └── 05_adversarial.ipynb    # FGSM 공격
├── src/
│   ├── dataset.py          # Dataset 클래스
│   └── model.py            # ResNet-50 모델
├── data/
│   └── train/
│       ├── shard_0/        # 이미지 + labels.csv
│       └── shard_1/
├── results/
│   ├── best_resnet50.pth   # 학습된 모델 가중치
│   ├── clean_indices.json  # Deduplication 결과
│   ├── training_curve.png
│   ├── confusion_matrix_clean.png
│   └── robustness_analysis.png
└── README.md
```

---

## 📦 Dataset

**NTIRE 2026 DeepFake Detection Dataset**
- 출처: [MSU Video Group](https://github.com/msu-video-group/NTIRE-2026-DeepFake-Detection)
- 전체 규모: 약 277,000장, 42개 생성 모델, 36개 왜곡 변환
- 본 프로젝트 사용: **72,112장** (Real 36,056 : AI 36,056, 1:1 균형)
- Train / Val / Test: **8 : 1 : 1** (57,689 / 7,212 / 7,211)

| 생성 모델 계열 | 모델 |
|:---|:---|
| Diffusion | Stable Diffusion 1.4/1.5/2.1, SDXL, PixArt, DeepFloyd IF, FLUX.1 등 |
| Transformer/Flow | Janus Pro 7B, Infinity, OmniGen, Ovis Image 등 |
| 기타 | Kandinsky, Kolors, YOSO 등 |

---

## 🚀 Quick Start

### 1. 환경 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pytorch-grad-cam scikit-learn imagehash tqdm Pillow pandas matplotlib
```

### 2. 데이터셋 준비

```
data/train/
├── shard_0/
│   ├── images/*.jpg
│   └── labels.csv
└── shard_1/
    ├── images/*.jpg
    └── labels.csv
```

### 3. 실행

```bash
# final_report.ipynb 위치: deepfake-detection/ 루트
# SKIP_TRAINING = False (처음 실행)
# SKIP_TRAINING = True  (재실행 시 학습 스킵)
jupyter notebook final_report.ipynb
```

---

## 🧠 Model

```
ResNet-50 (ImageNet pretrained)
    └── FC: Dropout(0.5) + Linear(2048, 2)
```

| 하이퍼파라미터 | 값 |
|:---|:---:|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-2 |
| Batch Size | 32 |
| Scheduler | Warmup(3) + CosineAnnealing |
| Early Stopping | Patience=5 (Val AUC) |

---

## 🔬 Key Findings

### Phase 1 — 탐지 성능
- Clean Test AUC **0.9546**, Accuracy **0.8667** 달성
- 20개 이상의 다양한 생성 모델이 포함된 데이터셋에서 높은 일반화 성능

### Phase 2 — Grad-CAM
- 모델이 AI 이미지 탐지 시 텍스처, 경계선, 배경 아티팩트에 집중
- FGSM 공격 후 attention이 분산되며 판단 근거 교란 시각적 확인

### Phase 3 — FGSM 견고성
- **AUC 단독 평가의 한계**: eps=0.01에서 AUC 0.0029로 붕괴했음에도 AUC 그래프만 보면 견고해 보임
- **ASR 0.9651** (eps=0.01): 단순한 단일 스텝 공격에도 모델이 완전히 무력화
- **결론**: AUC + Accuracy Drop + ASR 다중 지표 평가 필수

---

## 📚 References

| # | 논문 | 링크 |
|:---:|:---|:---:|
| [1] | He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 | [arxiv](https://arxiv.org/abs/1512.03385) |
| [2] | Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, ICLR 2015 | [arxiv](https://arxiv.org/abs/1412.6572) |
| [3] | Selvaraju et al., *Grad-CAM*, ICCV 2017 | [arxiv](https://arxiv.org/abs/1610.02391) |
| [4] | Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019 | [arxiv](https://arxiv.org/abs/1711.05101) |
| [5] | Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, ICLR 2017 | [arxiv](https://arxiv.org/abs/1608.03983) |

---

## 🤖 AI Tool Usage

본 프로젝트에서는 Claude(Anthropic)를 코드 디버깅, 구현 방식 검토, 보고서 문장 개선을 위한 보조 도구로 사용하였습니다.
데이터셋 선택, 실험 설계, 코드 실행, 결과 해석 및 최종 내용 검토는 팀원들이 직접 수행하였습니다.
