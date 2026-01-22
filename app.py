# -*- coding: utf-8 -*-
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import re
from datetime import datetime
from ultralytics import YOLO
import easyocr

# ==========================
# OCR Engine
# ==========================
class ProductionOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)

    def extract(self, img_crop, field_type):
        try:
            gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            if field_type == 'id_number':
                _, gray = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            results = self.reader.readtext(gray)
            if not results:
                return {'text': '', 'conf': 0.0}

            texts = [res[1] for res in results]
            confs = [res[2] for res in results]

            full_text = " ".join(texts)
            clean_text = self._clean(full_text, field_type)

            return {
                'text': clean_text,
                'conf': float(np.mean(confs)),
                'source': 'easyocr'
            }
        except Exception as e:
            return {'text': '', 'conf': 0.0}

    def _clean(self, text, field_type):
        if field_type == 'id_number':
            digits = re.findall(r'\d+', text)
            digits = ''.join(digits)
            return digits[-14:] if len(digits) >= 14 else digits

        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        return text.strip()


# ==========================
# ID Logic
# ==========================
class IDLogic:
    GOVS = {
        '01': 'القاهرة', '02': 'الإسكندرية', '03': 'بورسعيد', '04': 'السويس',
        '11': 'دمياط', '12': 'الدقهلية', '13': 'الشرقية', '14': 'القليوبية',
        '15': 'كفر الشيخ', '16': 'الغربية', '17': 'المنوفية', '18': 'البحيرة',
        '19': 'الإسماعاعيلية', '21': 'الجيزة', '22': 'بني سويف', '23': 'الفيوم',
        '24': 'المنيا', '25': 'أسيوط', '26': 'سوهاج', '27': 'قنا',
        '28': 'أسوان', '29': 'الأقصر', '31': 'البحر الأحمر',
        '32': 'الوادي الجديد', '33': 'مطروح',
        '34': 'شمال سيناء', '35': 'جنوب سيناء'
    }

    @staticmethod
    def parse(id_num):
        id_num = re.sub(r'\D', '', id_num)
        if len(id_num) != 14:
            return {'valid': False}

        try:
            century = int(id_num[0])
            year = (1900 if century == 2 else 2000) + int(id_num[1:3])
            month = int(id_num[3:5])
            day = int(id_num[5:7])
            gov_code = id_num[7:9]
            gender = 'ذكر' if int(id_num[12]) % 2 == 1 else 'أنثى'
            age = datetime.now().year - year

            return {
                'valid': True,
                'birth_date': f"{day:02d}/{month:02d}/{year}",
                'gender': gender,
                'age': age,
                'governorate': IDLogic.GOVS.get(gov_code, 'غير معروف'),
                'country': 'مصري 🇪🇬'
            }
        except:
            return {'valid': False}


# ==========================
# Pipeline
# ==========================
class Pipeline:
    field_map = {
        0: 'address',
        1: 'birth_date',
        2: 'country',
        3: 'full_name',
        4: 'id_number'
    }

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to('cpu')
        self.ocr = ProductionOCR()

    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return {'success': False, 'error': 'Image load failed'}

        results = self.model(img, device='cpu', conf=0.25, verbose=False)
        fields = {}

        for r in results:
            for box in r.boxes:
                cid = int(box.cls[0])
                field = self.field_map.get(cid)
                if not field:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pad = 15
                crop = img[
                    max(0, y1 - pad):min(img.shape[0], y2 + pad),
                    max(0, x1 - pad):min(img.shape[1], x2 + pad)
                ]

                ocr_res = self.ocr.extract(crop, field)

                fields[field] = {
                    'text': ocr_res['text'],
                    'confidence': min(float(box.conf[0]), ocr_res['conf']),
                    'bbox': [x1, y1, x2, y2]
                }

        id_info = None
        if 'id_number' in fields:
            id_info = IDLogic.parse(fields['id_number']['text'])

        return {
            'success': True,
            'fields': fields,
            'id_logic': id_info
        }


# ==========================
# Visualization
# ==========================
def draw_boxes(image, fields):
    img = image.copy()
    for k, v in fields.items():
        if 'bbox' not in v:
            continue
        x1, y1, x2, y2 = v['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, k, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return img


# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="Egyptian ID OCR", layout="wide")
st.title("🆔 OCR النظام القومي المصري")

uploaded_file = st.file_uploader("📤 ارفع صورة البطاقة", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_path = f"temp_{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    MODEL_PATH = "/home/marno000onaaa/Desktop/OCR_IDS/OCR_System_for_National_ID_Cards-Egyptian-Kuwaiti-/OCR_System_for_National_ID_Cards-Egyptian-Kuwaiti-/models/best.pt"

    pipe = Pipeline(MODEL_PATH)
    result = pipe.process(img_path)

    orig = cv2.imread(img_path)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)

    with col1:
        st.image(orig_rgb, caption="الصورة الأصلية", use_column_width=True)

    with col2:
        boxed = draw_boxes(orig, result['fields'])
        boxed_rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
        st.image(boxed_rgb, caption="YOLO Bounding Boxes", use_column_width=True)

    st.subheader("📋 البيانات المستخرجة")

    labels = {
        'full_name': 'الاسم',
        'id_number': 'الرقم القومي',
        'birth_date': 'تاريخ الميلاد',
        'address': 'العنوان'
    }

    for k, label in labels.items():
        if k in result['fields']:
            d = result['fields'][k]
            st.write(f"**{label}:** {d['text']} (ثقة {d['confidence']:.1%})")

    if result['id_logic'] and result['id_logic'].get('valid'):
        info = result['id_logic']
        st.subheader("🧠 معلومات مستخرجة من الرقم القومي")
        st.write(f"الجنسية: {info['country']}")
        st.write(f"تاريخ الميلاد: {info['birth_date']}")
        st.write(f"العمر: {info['age']} سنة")
        st.write(f"النوع: {info['gender']}")
        st.write(f"المحافظة: {info['governorate']}")
