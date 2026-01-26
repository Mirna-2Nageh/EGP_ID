import streamlit as st
import cv2
import numpy as np
import re
from datetime import datetime
from ultralytics import YOLO
import easyocr

# =====================================================
# Utils
# =====================================================
AR2EN = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')

def normalize_digits(text):
    return text.translate(AR2EN)

def only_digits(text):
    return re.sub(r'\D', '', text)

def clean_arabic(text):
    return re.sub(r'[^\u0600-\u06FF\s]', '', text).strip()

def extract_date(text):
    text = normalize_digits(text)
    m = re.search(r'(19|20)\d{2}[\-/\. ]?(0[1-9]|1[0-2])[\-/\. ]?(0[1-9]|[12][0-9]|3[01])', text)
    if not m:
        return ''
    s = m.group(0)
    s = re.sub(r'\D', '', s)
    return f"{s[6:8]}/{s[4:6]}/{s[0:4]}"

# =====================================================
# OCR Engine (Field-aware)
# =====================================================
class ProductionOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)

    def extract(self, img, field):
        if img is None or img.size == 0:
            return {'text': '', 'conf': 0.0}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        if field == 'id_number':
            return self._id_ocr(gray)
        if field == 'birth_date':
            return self._date_ocr(gray)
        if field == 'full_name':
            return self._name_ocr(gray)
        if field == 'address':
            return self._address_ocr(gray)

        return {'text': '', 'conf': 0.0}

    # ---------------- ID NUMBER ----------------
    def _id_ocr(self, gray):
        """
        Enhanced ID extraction using multi-thresholding and length validation.
        """
        # 1. Multi-Pass Preprocessing
        # We try 3 different looks at the same ID area to see which one the OCR likes best
        results_pool = []
        
        # Pass A: High Contrast Bilateral (Preserves edges, removes noise)
        pass_a = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Pass B: Adaptive Thresholding (Good for uneven lighting/shadows)
        pass_b = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 10
        )

        # Pass C: Morphological Opening (Removes small "salt and pepper" noise)
        kernel = np.ones((2,2), np.uint8)
        pass_c = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        variants = [gray, pass_a, pass_b, pass_c]

        for v in variants:
            # allowlist forces the OCR to ignore letters entirely
            res = self.reader.readtext(v, allowlist='0123456789٠١٢٣٤٥٦٧٨٩', detail=1)
            if not res:
                continue
                
            # Combine all fragments found in one image
            text = "".join([normalize_digits(r[1]) for r in res])
            digits_only = only_digits(text)
            
            # Confidence score (average of all detected fragments)
            avg_conf = np.mean([r[2] for r in res]) if res else 0
            
            # Weighted Scoring: We prioritize detections that are exactly 14 digits
            score = avg_conf
            if len(digits_only) == 14:
                score += 0.5  # Heavy boost for correct length
            elif len(digits_only) > 14:
                digits_only = digits_only[:14] # Trim if redundant info caught
            
            results_pool.append({'text': digits_only, 'score': score, 'conf': avg_conf})

        # 2. Selection Logic (The "Voting" mechanism)
        if not results_pool:
            return {'text': '', 'conf': 0.0}

        # Sort by the score (Confidence + Length Bonus)
        best_match = max(results_pool, key=lambda x: x['score'])
        
        return {'text': best_match['text'], 'conf': best_match['conf']}
    # ---------------- DATE ----------------
    def _date_ocr(self, gray):
        res = self.reader.readtext(gray, detail=1)
        text = ' '.join([r[1] for r in res if len(r) > 1])
        date = extract_date(text)
        confs = [r[2] for r in res if len(r) > 2]
        conf = np.mean(confs) if confs else 0
        return {'text': date, 'conf': conf}

    # ---------------- NAME ----------------
    def _name_ocr(self, gray):
        res = self.reader.readtext(gray, paragraph=True, detail=1)
        text = ' '.join([r[1] for r in res if len(r) > 1])
        confs = [r[2] for r in res if len(r) > 2]
        return {'text': clean_arabic(text), 'conf': np.mean(confs) if confs else 0}

    # ---------------- ADDRESS ----------------
    def _address_ocr(self, gray):
        res = self.reader.readtext(gray, paragraph=True, detail=1)
        text = ' '.join([r[1] for r in res if len(r) > 1])
        confs = [r[2] for r in res if len(r) > 2]
        return {'text': normalize_digits(clean_arabic(text)), 'conf': np.mean(confs) if confs else 0}

# =====================================================
# ID Logic
# =====================================================
class IDLogic:
    @staticmethod
    def parse(nid):
        nid = only_digits(nid)
        if len(nid) != 14:
            return {'valid': False}
        try:
            century = int(nid[0])
            year = (1900 if century == 2 else 2000) + int(nid[1:3])
            month = int(nid[3:5])
            day = int(nid[5:7])
            gender = 'ذكر' if int(nid[12]) % 2 else 'أنثى'
            age = datetime.now().year - year
            return {
                'valid': True,
                'birth_date': f"{day:02d}/{month:02d}/{year}",
                'gender': gender,
                'age': age,
                'country': 'مصري 🇪🇬'
            }
        except:
            return {'valid': False}

# =====================================================
# Pipeline
# =====================================================
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
            raise ValueError(f"Image not found or failed to load: {img_path}")

        draw_img = img.copy()  # 👈 نسخة للرسم

        results = self.model(img, conf=0.25, device='cpu', verbose=False)

        fields = {}

        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                field = self.field_map.get(cls_idx)
                if not field:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # 🔲 رسم البوكس
                cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    draw_img,
                    f"{field} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                crop = img[max(0,y1-10):y2+10, max(0,x1-10):x2+10]
                if crop is None or crop.size == 0:
                    continue

                ocr = self.ocr.extract(crop, field)
                fields[field] = {
                    'text': ocr['text'],
                    'confidence': min(conf, ocr['conf'])
                }

        id_logic = IDLogic.parse(fields['id_number']['text']) if 'id_number' in fields else None

        return fields, id_logic, img, draw_img


# =====================================================
# Streamlit
# =====================================================
st.set_page_config("Egyptian ID OCR", layout="wide")
st.title("🆔 Egyptian National ID OCR – Enhanced")

file = st.file_uploader("📤 ارفع صورة البطاقة", type=["jpg","png","jpeg"])

if file:
    path = f"temp_{file.name}"
    with open(path,"wb") as f:
        f.write(file.getbuffer())

    pipe = Pipeline("OCR_System_for_National_ID_Cards-Egyptian-Kuwaiti-/models/best.pt")
    fields, id_logic, original_img, boxed_img = pipe.process(path)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("📦 Detection Result")
        st.image(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB), use_container_width=True)



    for k, v in fields.items():
        st.write(f"**{k}:** {v['text']} ({v['confidence']:.1%})")

    if id_logic and id_logic['valid']:
        st.success("✅ الرقم القومي صحيح")
        for k, v in id_logic.items():
            if k != 'valid':
                st.write(f"{k}: {v}")
