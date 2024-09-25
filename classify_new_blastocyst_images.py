import os
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
from scipy.stats import entropy  # 엔트로피 함수 임포트


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ------------------------ Configuration ------------------------

# Path to the directory containing new blastocyst images
NEW_IMAGE_DIR = r"C:\Users\vince\Downloads\viability"  # Update this path as needed

# Path to the saved RandomForestClassifier and Scaler
MODEL_DIR = r"C:\Users\vince\Documents\my_models"  # Update if different
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Define image size for InceptionV3
IMG_SIZE = (299, 299)

# Path to save the classification results
RESULTS_CSV_PATH = os.path.join(NEW_IMAGE_DIR, 'classification_results.csv')

# Probability threshold for marking images for review
PROB_THRESHOLD = 0.4  # 임계값 설정 (0.4으로 변경)

# Entropy threshold for marking images for review
ENTROPY_THRESHOLD = 1.4  # 엔트로피 임계값 설정 (1.4으로 변경)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------ Load Models ------------------------

# Check if model files exist
if not os.path.exists(RANDOM_FOREST_MODEL_PATH):
    raise FileNotFoundError(f"RandomForestClassifier 모델을 찾을 수 없습니다: '{RANDOM_FOREST_MODEL_PATH}'")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler 파일을 찾을 수 없습니다: '{SCALER_PATH}'")

# Load Inception-v3 model with aux_logits=True
inception_model = models.inception_v3(pretrained=True, aux_logits=True)
# Remove the final fully connected layer
inception_model.fc = torch.nn.Identity()
# Remove the auxiliary logits
inception_model.AuxLogits = torch.nn.Identity()
inception_model.to(DEVICE)
inception_model.eval()

# Load ViT 모델
from transformers import ViTForImageClassification, ViTFeatureExtractor, ViTConfig

config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=5)
model_vit = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    config=config,
    ignore_mismatched_sizes=True  # 크기 불일치 무시
)
model_vit.to(DEVICE)
model_vit.eval()

# Load Feature Extractor for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Load RandomForestClassifier and Scaler
with open(RANDOM_FOREST_MODEL_PATH, 'rb') as f:
    rf_classifier = joblib.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = joblib.load(f)

# ------------------------ Define Transforms ------------------------

# Define the image transformations: resize, center crop, to tensor, normalize
preprocess_inception = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet 평균값
        std=[0.229, 0.224, 0.225]    # ImageNet 표준편차
    )
])

# ------------------------ Helper Functions ------------------------

import cv2
from skimage import measure

def extract_morphological_features(image_path):
    """
    이미지에서 형태학적 특징을 추출합니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로.
    
    Returns:
        list: [blastocyst_area, icm_area, circularity, blastocyst_density, perimeter_to_area_ratio]
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return [0, 0, 0, 0, 0]
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 이진화
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 라벨링을 통해 객체 분할
    labels = measure.label(thresh, connectivity=2)
    properties = measure.regionprops(labels)

    # 배반포 전체 영역 추출
    if properties:
        # 가장 큰 객체를 배반포로 간주
        largest_contour = max(properties, key=lambda x: x.area)
        blastocyst_area = largest_contour.area
        blastocyst_perimeter = largest_contour.perimeter

        # TE 품질: 원형도 계산
        circularity = (4 * np.pi * blastocyst_area) / (blastocyst_perimeter ** 2) if blastocyst_perimeter != 0 else 0

        # ICM 추출: 배반포 내부의 가장 큰 객체를 ICM으로 간주
        mask = labels == largest_contour.label
        masked_image = np.where(mask, gray, 0)
        _, inner_thresh = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inner_labels = measure.label(inner_thresh, connectivity=2)
        inner_properties = measure.regionprops(inner_labels)

        if inner_properties:
            icm = max(inner_properties, key=lambda x: x.area)
            icm_area = icm.area
        else:
            icm_area = 0

        # 추가적인 형태학적 특징 계산
        blastocyst_density = icm_area / blastocyst_area if blastocyst_area != 0 else 0
        perimeter_to_area_ratio = blastocyst_perimeter / blastocyst_area if blastocyst_area != 0 else 0

        return [blastocyst_area, icm_area, circularity, blastocyst_density, perimeter_to_area_ratio]
    else:
        # 객체가 없을 경우 0으로 채움
        return [0, 0, 0, 0, 0]

def extract_inception_features(image_path, model, preprocess, device):
    """
    InceptionV3를 사용하여 이미지에서 딥러닝 특징을 추출합니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로.
        model (torch.nn.Module): InceptionV3 모델.
        preprocess (torchvision.transforms.Compose): 전처리 변환.
        device (torch.device): 디바이스.
    
    Returns:
        np.ndarray: 추출된 특징 벡터.
    """
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스 이동

    with torch.no_grad():
        # 중간 레이어의 출력을 얻기 위해 hook 사용
        features = []

        def hook(module, input, output):
            features.append(output.cpu().numpy())

        # 'avgpool' 레이어에 hook 등록
        handle = model.avgpool.register_forward_hook(hook)
        model(input_batch)
        handle.remove()

    if features:
        features_array = features[0].flatten()
        return features_array
    else:
        return np.array([])

def extract_vit_features(image_path, model_vit, feature_extractor, device):
    """
    ViT를 사용하여 이미지에서 딥러닝 특징을 추출합니다.
    
    Parameters:
        image_path (str): 이미지 파일 경로.
        model_vit (transformers.ViTForImageClassification): ViT 모델.
        feature_extractor (transformers.ViTFeatureExtractor): ViT 특징 추출기.
        device (torch.device): 디바이스.
    
    Returns:
        np.ndarray: 추출된 특징 벡터.
    """
    image = Image.open(image_path).convert('RGB')
    encoding = feature_extractor(images=image, return_tensors='pt')
    input_ids = encoding['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model_vit(input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        features = probs.cpu().numpy().flatten()

    return features

# ------------------------ Classification ------------------------

def classify_new_images():
    """
    라벨이 없는 새로운 배반포 이미지를 분류하고 결과를 CSV 파일로 저장합니다.
    
    Returns:
        None
    """
    # 디렉토리 내의 PNG, JPG, JPEG 파일 목록 가져오기
    image_files = [f for f in os.listdir(NEW_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"디렉토리에 이미지 파일이 없습니다: {NEW_IMAGE_DIR}")
        return

    # 결과를 저장할 리스트 초기화
    results = []

    print(f"총 {len(image_files)}개의 이미지를 찾았습니다. 분류를 시작합니다...")

    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(NEW_IMAGE_DIR, img_file)
        
        if idx % 100 == 0 and idx != 0:
            print(f"처리 중: {idx}/{len(image_files)}")

        # 1. 형태학적 특징 추출
        morph_features = extract_morphological_features(img_path)

        # 2. InceptionV3 특징 추출
        inception_features = extract_inception_features(img_path, inception_model, preprocess_inception, DEVICE)
        if inception_features.size == 0:
            # 딥러닝 특징이 추출되지 않은 경우 0으로 채움
            inception_features = np.zeros(2048)  # InceptionV3의 avgpool 출력 크기 (2048)

        # 3. ViT 특징 추출
        vit_features = extract_vit_features(img_path, model_vit, feature_extractor, DEVICE)
        if vit_features.size == 0:
            # 딥러닝 특징이 추출되지 않은 경우 0으로 채움
            vit_features = np.zeros(5)  # ViT의 num_labels=5

        # 4. 특징 결합
        combined_features = list(morph_features) + list(inception_features) + list(vit_features)

        # 5. 특징 스케일링
        combined_features_scaled = scaler.transform([combined_features])

        # 6. 예측 클래스 확률
        probabilities = rf_classifier.predict_proba(combined_features_scaled)[0]

        # 7. 예측 클래스
        predicted_class = rf_classifier.predict(combined_features_scaled)[0]

        # 8. 확률 임계값과 엔트로피 기준에 따라 'Reviewed' 여부 결정
        max_prob = np.max(probabilities)
        image_entropy = entropy(probabilities)
        if max_prob > PROB_THRESHOLD or image_entropy > ENTROPY_THRESHOLD:
            reviewed = True
        else:
            reviewed = False

        # 9. 결과 저장
        results.append({
            'Image': img_file,
            'Predicted_Class': predicted_class,
            'Probability_Class_1': probabilities[0],
            'Probability_Class_2': probabilities[1],
            'Probability_Class_3': probabilities[2],
            'Probability_Class_4': probabilities[3],
            'Probability_Class_5': probabilities[4],
            'Entropy': image_entropy,  # 엔트로피 추가
            'Reviewed': reviewed  # 검토 필요 여부 추가
        })

        print(f"Classified {img_file}: Class {predicted_class} with probabilities {probabilities}, Entropy: {image_entropy:.2f}, Reviewed: {reviewed}")

    # 10. 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)

    # 11. CSV 파일로 저장
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\n분류가 완료되었습니다. 결과가 '{RESULTS_CSV_PATH}'에 저장되었습니다.")

    # 12. 검토가 필요한 이미지 별도 저장 (옵션)
    reviewed_df = results_df[results_df['Reviewed'] == True]
    reviewed_csv_path = os.path.join(NEW_IMAGE_DIR, 'images_for_review.csv')
    reviewed_df.to_csv(reviewed_csv_path, index=False)
    print(f"검토가 필요한 이미지가 '{reviewed_csv_path}'에 저장되었습니다.")

if __name__ == "__main__":
    classify_new_images()