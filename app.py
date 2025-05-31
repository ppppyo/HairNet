import streamlit as st
from PIL import Image
import torch
import json
from torchvision import transforms
from models.cnn_model import CNNModel
# from grad_cam import GradCAM, overlay_heatmap_on_image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision.transforms import ToPILImage
from utils import preprocess_image, load_model,inference

st.set_page_config(page_title="헤어스타일 분류기", layout="centered")

st.title("✨ HairNet ✨")

for key in ["gender", "style_type"]:
    if key not in st.session_state:
        st.session_state[key]=None

with open("style_descriptions.json", "r", encoding="utf-8") as f:
    STYLE_DESCRIPTION=json.load(f)

# 첫화면, 성별 선택
if st.session_state.gender is None:
    st.markdown("### 👉 성별을 선택해주세요")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨 남자"):
            st.session_state.gender = "male"
            st.rerun()
    with col2:
        if st.button("👩 여자"):
            st.session_state.gender = "female"
            st.rerun()

# 두번째, 스타일 선택(펌/컷)
elif st.session_state.style_type is None:
    st.markdown(f"### 👉 { '남성' if st.session_state.gender == 'male' else '여성' } 헤어스타일 유형을 선택해주세요")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✂️ 컷 스타일"):
            st.session_state.style_type = "cut"
            st.rerun()
    with col2:
        if st.button("💈 펌 스타일"):
            st.session_state.style_type = "perm"
            st.rerun()            

# 세번째, 이미지 업로드 페이지
else:
    gender = st.session_state.gender
    style_type = st.session_state.style_type
    combo_key = f"{gender}_{style_type}"

    CLASS_MAP= {
        "male_cut":["크롭컷", "댄디컷", "필러스컷","가일컷", "포마드컷", "리젠트컷"],
        "male_perm":["애즈펌", "히피펌", "리프펌", "쉐도우펌", "스왈로펌"],
        "female_cut":["히메컷", "허쉬컷", "레이어드컷", "모즈컷", "슬릭컷", "테슬컷"],
        "female_perm":["빌드펌", "구름펌", "그레이스펌", "히피펌"]
    }
    MODEL_PATHS={
        "male_cut":"best_model/male_cut.pth",
        "male_perm":"best_model/male_perm.pth",
        "female_cut":"best_model/female_cut.pth",
        "female_perm":"best_model/female_perm.pth"
    }
    style_kor = {
        "male": "👨 남성",
        "female": "👩 여성",
        "cut": "컷 스타일",
        "perm": "펌 스타일"
    }

    gender_kor = style_kor.get(gender, "")
    style_type_kor = style_kor.get(style_type, "")

    # 안내 문구 출력
    st.markdown(f"### {gender_kor} {style_type_kor} 이미지를 업로드해주세요!")

    CLASSES=CLASS_MAP[combo_key]
    model_path=MODEL_PATHS[combo_key]
    num_classes =len(CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = load_model(model_path, device, num_classes)
    # target_layer=model.features[-2]
    
    #파일 업로드 
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="업로드된 이미지", use_container_width=True)
        #이미지 -> Tensor로 변환    
        input_tensor = preprocess_image(image)
        
        #에러메시지
        if input_tensor is None:
            st.error("❗ 얼굴이 하나만 나온 사진을 입력해주세요.")
        else:   
            input_tensor.to(device)
            #모델에 인풋텐서를 입력
            pred_class, result = inference(input_tensor, model_path, num_classes, CLASSES)

            st.markdown(f"### 헤어스타일 💇 **{pred_class}** ({result}%)")
            
            #예측결과 해당하는 스타일 설명 불러옴
            if pred_class in STYLE_DESCRIPTION:
                description = STYLE_DESCRIPTION[pred_class].replace("\n", "<br>")
                st.markdown(f"""
                <div style='
                    font-family: "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
                    font-size: 16px;
                    line-height: 1.6;
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-radius: 10px;
                    border: 1px solid #ddd;
                '>
                <strong>💬 스타일 설명</strong><br>{description}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("&nbsp;", unsafe_allow_html=True)
            if st.button("처음으로"):
                st.session_state.gender =None
                st.session_state.style_type =None
                st.rerun()
            #plot_probabilities(result, CLASSES)
            
            # topil=ToPILImage()
            # st.image(topil(input_tensor.squeeze(0)))
            # gradcam=GradCAM(model, target_layer)
            # heatmap=gradcam.generate(input_tensor)
            # overlay = overlay_heatmap_on_image(heatmap,image)
            # st.image(overlay, caption="Grad-cam", use_container_width=True)
            # ({probs.max().item()*100:.2f}%)      
