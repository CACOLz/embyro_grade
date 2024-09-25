import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.stats import entropy
from transformers import ViTForImageClassification, ViTImageProcessor, ViTConfig
import cv2
from skimage import measure
import os

# ------------------------ Configuration ------------------------

# 현재 파일의 디렉토리 경로 얻기
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the saved RandomForestClassifier and Scaler
RANDOM_FOREST_MODEL_PATH = os.path.join(BASE_DIR, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Probability and Entropy thresholds
PROB_THRESHOLD = 0.4
ENTROPY_THRESHOLD = 1.4

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------ Load Models ------------------------

@st.cache_resource
def load_models():
    # Check if model files exist
    if not os.path.exists(RANDOM_FOREST_MODEL_PATH):
        st.error(f"RandomForestClassifier 모델을 찾을 수 없습니다: '{RANDOM_FOREST_MODEL_PATH}'")
        return None, None, None, None, None
    
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler 파일을 찾을 수 없습니다: '{SCALER_PATH}'")
        return None, None, None, None, None

    # Load Inception-v3 model with weights
    inception_weights = models.Inception_V3_Weights.IMAGENET1K_V1
    inception_model = models.inception_v3(weights=inception_weights, aux_logits=True)
    inception_model.fc = torch.nn.Identity()
    inception_model.AuxLogits = torch.nn.Identity()
    inception_model.to(DEVICE)
    inception_model.eval()

    # Load ViT 모델
    config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=5)
    model_vit = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        config=config,
        ignore_mismatched_sizes=True
    )
    model_vit.to(DEVICE)
    model_vit.eval()

    # Load Feature Extractor for ViT
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Load RandomForestClassifier and Scaler
    rf_classifier = joblib.load(RANDOM_FOREST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return inception_model, model_vit, feature_extractor, rf_classifier, scaler

# Load models
inception_model, model_vit, feature_extractor, rf_classifier, scaler = load_models()

if inception_model is None or model_vit is None or feature_extractor is None or rf_classifier is None or scaler is None:
    st.stop()

# ------------------------ Define Transforms ------------------------

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

def extract_morphological_features(image):
    """
    이미지에서 형태학적 특징을 추출합니다.
    
    Parameters:
        image (PIL.Image): 이미지 객체.
    
    Returns:
        list: [blastocyst_area, icm_area, circularity, blastocyst_density, perimeter_to_area_ratio]
    """
    # PIL 이미지를 OpenCV 형식으로 변환
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if image_cv is None:
        st.warning("이미지를 로드할 수 없습니다.")
        return [0, 0, 0, 0, 0]
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
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

def extract_inception_features(image, model, preprocess, device):
    """
    InceptionV3를 사용하여 이미지에서 딥러닝 특징을 추출합니다.
    
    Parameters:
        image (PIL.Image): 이미지 객체.
        model (torch.nn.Module): InceptionV3 모델.
        preprocess (torchvision.transforms.Compose): 전처리 변환.
        device (torch.device): 디바이스.
    
    Returns:
        np.ndarray: 추출된 특징 벡터.
    """
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

def extract_vit_features(image, model_vit, feature_extractor, device):
    """
    ViT를 사용하여 이미지에서 딥러닝 특징을 추출합니다.
    
    Parameters:
        image (PIL.Image): 이미지 객체.
        model_vit (transformers.ViTForImageClassification): ViT 모델.
        feature_extractor (transformers.ViTImageProcessor): ViT 특징 추출기.
        device (torch.device): 디바이스.
    
    Returns:
        np.ndarray: 추출된 특징 벡터.
    """
    encoding = feature_extractor(images=image, return_tensors='pt')
    input_ids = encoding['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model_vit(input_ids)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        features = probs.cpu().numpy().flatten()

    return features

# ------------------------ Streamlit App ------------------------

def main():
    st.title("Blastocyst Grade Classification")
    st.write("이미지를 업로드하면 자동으로 Grade를 예측합니다. 불확실한 예측은 검토 대상으로 표시됩니다.")

    uploaded_files = st.file_uploader("이미지 파일 업로드", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if uploaded_files:
        results = []

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)

            # 1. 형태학적 특징 추출
            morph_features = extract_morphological_features(image)

            # 2. InceptionV3 특징 추출
            inception_features = extract_inception_features(image, inception_model, preprocess_inception, DEVICE)
            if inception_features.size == 0:
                inception_features = np.zeros(2048)

            # 3. ViT 특징 추출
            vit_features = extract_vit_features(image, model_vit, feature_extractor, DEVICE)
            if vit_features.size == 0:
                vit_features = np.zeros(5)

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
            result = {
                'Image': uploaded_file.name,
                'Predicted_Class': predicted_class,
                'Probability_Class_1': probabilities[0],
                'Probability_Class_2': probabilities[1],
                'Probability_Class_3': probabilities[2],
                'Probability_Class_4': probabilities[3],
                'Probability_Class_5': probabilities[4],
                'Entropy': image_entropy,
                'Reviewed': reviewed
            }
            results.append(result)

            # 10. 결과 표시
            st.markdown(f"**Predicted Class:** {predicted_class}")
            st.markdown(f"**Probabilities:**")
            st.write(pd.DataFrame(probabilities.reshape(1, -1), columns=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))
            st.markdown(f"**Entropy:** {image_entropy:.2f}")
            if reviewed:
                st.warning("🔴 이 이미지는 검토가 필요합니다.")
            else:
                st.success("🟢 이 이미지는 정상적으로 분류되었습니다.")
            st.markdown("---")

        # 11. 전체 결과를 CSV로 다운로드
        if st.button("전체 결과 다운로드"):
            results_df = pd.DataFrame(results)
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="CSV 파일 다운로드",
                data=csv,
                file_name='classification_results.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()
