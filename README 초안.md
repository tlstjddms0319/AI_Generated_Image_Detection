# **Deepfake Detection Project**

---

## **프로젝트 소개**

최근 생성형 AI 모델의 성능이 빠르게 발전하면서 AI가 생성한 이미지와 실제 이미지를 구분하는 것이 점점 어려워지고 있다.

이번 프로젝트에서는 ResNet-50 기반 이미지 분류 모델을 이용해 AI 생성 이미지 탐지를 진행하였고 Grad-CAM을 활용해 모델이 어떤 부분을 보고 판단하는지 시각화하였다.

또한 FGSM(Fast Gradient Sign Method) 기반 적대적 공격을 적용하여 작은 노이즈에도 모델 성능이 얼마나 영향을 받는지 함께 확인하였다.

단순히 분류 정확도만 확인하는 것이 아니라 모델이 실제로 어떤 특징을 학습하는지, 적대적 공격에 얼마나 취약한지를 함께 분석해보는 것을 목표로 프로젝트를 진행하였다.

---

## **개발 환경**

* Python 3.11  
* PyTorch  
* Jupyter Notebook  
* Google Colab (GPU 환경 사용)

---

## **데이터셋**

NTIRE 2026 Robust AI-Generated Image Detection Challenge에서 공개된 데이터셋을 기반으로 실험을 진행하였다.

원본 데이터셋은 실제 이미지와 AI 생성 이미지를 포함하고 있으며 다양한 생성 모델과 실제 환경 변형(in-the-wild transformation)이 반영되어 있다.

프로젝트에서는 Hugging Face에 공개된 train 데이터셋 중 일부 폴더만 사용하여 실험을 진행하였다.

사용 데이터:

* Real 이미지 약 5,000장  
* AI Generated 이미지 약 5,000장

총 10,000장의 이미지를 학습 및 평가에 사용하였다.

데이터셋 출처:

* [https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train)

### **데이터 증강**

다음과 같은 augmentation 기법을 사용하였다.

* Random Horizontal Flip  
* Random Rotation  
* ColorJitter

---

## **모델 학습**

### **사용 모델**

* ResNet-50 (ImageNet pretrained)

초기에는 간단한 CNN 구조도 실험했지만,  
성능 차이가 커서 최종적으로 ResNet-50 모델을 사용하였다.

### **학습 설정**

Optimizer : AdamW  
Learning Rate : 3e-4  
Batch Size : 32  
Loss Function : CrossEntropyLoss

Validation 성능 기준으로 Early Stopping을 적용하였다.

평가 지표는 ROC AUC를 사용하였다.

---

## **Grad-CAM 시각화**

Grad-CAM을 이용하여 모델이 실제로 이미지의 어느 부분을 중요하게 판단하는지 확인하였다.

실험 결과 일부 이미지에서는 얼굴 주변 texture나 경계 영역에 attention이 집중되는 경향을 확인할 수 있었다.

또한 적대적 공격 이후에는 attention 영역이 흐트러지는 모습도 일부 확인되었다.

---

## **Adversarial Attack**

FGSM(Fast Gradient Sign Method)을 이용해 적대적 공격 실험을 진행하였다.

작은 epsilon 값에서도 모델 성능이 크게 감소하는 현상이 나타났으며 이를 통해 이미지 탐지 모델이 적대적 공격에 상당히 취약할 수 있다는 점을 확인할 수 있었다.

### **실험 결과**

| Epsilon | ROC AUC |
| ----- | ----- |
| Clean | 0.9897 |
| 0.01 | 0.0401 |
| 0.03 | 0.0051 |
| 0.05 | 0.0057 |
| 0.10 | 0.0166 |

특히 epsilon=0.01 수준의 작은 노이즈에서도 성능이 급격하게 감소하는 현상이 나타났다.

---

## **프로젝트 구조**

deepfake-detection/  
├── notebooks/  
│   ├── 00\_setup\_check.ipynb  
│   ├── 01\_dataset.ipynb  
│   ├── 02\_deduplication.ipynb  
│   ├── 03\_train.ipynb  
│   ├── 04\_gradcam.ipynb  
│   └── 05\_adversarial.ipynb  
│  
├── src/  
│   ├── dataset.py  
│   └── model.py  
│  
├── results/  
├── requirements.txt  
└── README.md

---

## **실행 방법**

### **1\. 저장소 클론**

git clone \<repository\_url\>  
cd deepfake-detection

### **2\. 패키지 설치**

pip install \-r requirements.txt

### **3\. 노트북 실행**

다음 순서대로 notebook 파일을 실행하면 된다.

00\_setup\_check.ipynb  
01\_dataset.ipynb  
02\_deduplication.ipynb  
03\_train.ipynb  
04\_gradcam.ipynb  
05\_adversarial.ipynb

Google Colab GPU 환경에서 실행하는 것을 권장한다.

---

## **결과 파일**

results/  
├── training\_curve.png  
├── robustness\_analysis.png  
├── duplicate\_examples.png  
├── hashes.csv  
└── attack\_gradcam\_sample\*.png

---

## **한계점 및 개선 방향**

### **한계점**

* 적대적 공격에 대한 robustness가 낮음  
* 데이터셋 종류가 제한적임  
* 특정 생성 모델에 과적합될 가능성이 존재

### **개선 방향**

* 다양한 생성 모델 데이터 추가  
* ViT 기반 모델 실험  
* Adversarial Training 적용  
* 더 다양한 공격 기법(PGD 등) 실험

---

## **참고 문헌**

* Deep Residual Learning for Image Recognition (CVPR 2016\)  
* Grad-CAM: Visual Explanations from Deep Networks (ICCV 2017\)  
* Explaining and Harnessing Adversarial Examples (ICLR 2014\)

---

## **AI Tool Usage**

프로젝트 진행 과정에서 Gemini를 보조 도구로 활용하였다.

주로 Grad-CAM 관련 자료 조사, 데이터 증강 기법 탐색, 논문 검색 등에 활용하였으며  
최종 구현 및 코드 수정, 실험 진행은 직접 수행하였다.

