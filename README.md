# Blastocyst Grade Classification

이 프로젝트는 RandomForestClassifier와 딥러닝 특징(InceptionV3, ViT)을 사용하여 배반포 이미지의 품질을 분류하는 모델입니다. Streamlit을 사용하여 웹 애플리케이션 형태로 배포되었습니다.

## 기능

- 이미지 업로드를 통해 배반포의 품질 등급 예측
- 예측 확률과 엔트로피 기반의 검토 대상 표시
- 분류 결과를 CSV 파일로 다운로드 가능

## 사용 방법

1. **Streamlit 애플리케이션 실행:**

   ```bash
   streamlit run blastocyst_classifier_app.py
