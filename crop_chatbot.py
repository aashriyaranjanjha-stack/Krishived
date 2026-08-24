import pandas as pd
import streamlit as st
import pytesseract
import cv2
import re
import numpy as np

from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from deep_translator import GoogleTranslator


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Krishived",
    page_icon="🌾",
    layout="centered"
)


# =========================================================
# LANGUAGES
# =========================================================

lang_codes = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Bengali": "bn"
}


# =========================================================
# LABELS
# =========================================================

labels = {

    "English": {
        "title": "Krishived",
        "upload": "📎 Upload KVK / Soil Health Card",
        "upload_help": "Upload a clear image of your Soil Health Card",
        "extracted": "📄 Information extracted from your report",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "ph": "pH",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "rainfall": "Rainfall (mm)",
        "manual": "🌱 Enter other information",
        "button": "Recommend Crops",
        "crop_title": "🌾 Top 3 Recommended Crops",
        "not_detected": "Not detected",
        "uploaded": "Uploaded Soil Health Card"
    },

    "Hindi": {
        "title": "कृषिवेद",
        "upload": "📎 KVK / मृदा स्वास्थ्य कार्ड अपलोड करें",
        "upload_help": "अपने मृदा स्वास्थ्य कार्ड की स्पष्ट तस्वीर अपलोड करें",
        "extracted": "📄 आपकी रिपोर्ट से प्राप्त जानकारी",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फॉस्फोरस (P)",
        "potassium": "पोटैशियम (K)",
        "ph": "pH",
        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "rainfall": "वर्षा (mm)",
        "manual": "🌱 अन्य जानकारी दर्ज करें",
        "button": "फसल सुझाएं",
        "crop_title": "🌾 शीर्ष 3 सुझाई गई फसलें",
        "not_detected": "पता नहीं चला",
        "uploaded": "अपलोड किया गया मृदा स्वास्थ्य कार्ड"
    },

    "Kannada": {
        "title": "ಕೃಷಿವೇದ",
        "upload": "📎 KVK / ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "upload_help": "ನಿಮ್ಮ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್‌ನ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",
        "extracted": "📄 ನಿಮ್ಮ ವರದಿಯಿಂದ ಪಡೆದ ಮಾಹಿತಿ",
        "nitrogen": "ನೈಟ್ರಜನ (N)",
        "phosphorus": "ಫಾಸ್ಫರಸ್ (P)",
        "potassium": "ಪೊಟ್ಯಾಸಿಯಮ್ (K)",
        "ph": "pH",
        "temperature": "ತಾಪಮಾನ (°C)",
        "humidity": "ಆರ್ದ್ರತೆ (%)",
        "rainfall": "ಮಳೆ (mm)",
        "manual": "🌱 ಇತರ ಮಾಹಿತಿಯನ್ನು ನಮೂದಿಸಿ",
        "button": "ಬೆಳೆಗಳನ್ನು ಸೂಚಿಸಿ",
        "crop_title": "🌾 ಶಿಫಾರಸು ಮಾಡಲಾದ ಟಾಪ್ 3 ಬೆಳೆಗಳು",
        "not_detected": "ಪತ್ತೆಯಾಗಿಲ್ಲ",
        "uploaded": "ಅಪ್ಲೋಡ್ ಮಾಡಿದ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್"
    },

    "Punjabi": {
        "title": "ਕ੍ਰਿਸ਼ਿਵੇਦ",
        "upload": "📎 KVK / ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਅਪਲੋਡ ਕਰੋ",
        "upload_help": "ਆਪਣੇ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਦੀ ਸਾਫ਼ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",
        "extracted": "📄 ਤੁਹਾਡੀ ਰਿਪੋਰਟ ਤੋਂ ਪ੍ਰਾਪਤ ਜਾਣਕਾਰੀ",
        "nitrogen": "ਨਾਈਟ੍ਰੋਜਨ (N)",
        "phosphorus": "ਫਾਸਫੋਰਸ (P)",
        "potassium": "ਪੋਟਾਸ਼ੀਅਮ (K)",
        "ph": "pH",
        "temperature": "ਤਾਪਮਾਨ (°C)",
        "humidity": "ਨਮੀ (%)",
        "rainfall": "ਵਰਖਾ (mm)",
        "manual": "🌱 ਹੋਰ ਜਾਣਕਾਰੀ ਦਰਜ ਕਰੋ",
        "button": "ਫਸਲਾਂ ਦੀ ਸਿਫਾਰਸ਼ ਕਰੋ",
        "crop_title": "🌾 ਸਿਖਰ ਦੀਆਂ 3 ਸਿਫਾਰਸ਼ੀ ਫਸਲਾਂ",
        "not_detected": "ਪਤਾ ਨਹੀਂ ਲੱਗਿਆ",
        "uploaded": "ਅਪਲੋਡ ਕੀਤਾ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ"
    },

    "Bengali": {
        "title": "কৃষিবেদ",
        "upload": "📎 KVK / মাটি স্বাস্থ্য কার্ড আপলোড করুন",
        "upload_help": "আপনার মাটি স্বাস্থ্য কার্ডের একটি পরিষ্কার ছবি আপলোড করুন",
        "extracted": "📄 আপনার রিপোর্ট থেকে প্রাপ্ত তথ্য",
        "nitrogen": "নাইট্রোজেন (N)",
        "phosphorus": "ফসফরাস (P)",
        "potassium": "পটাশিয়াম (K)",
        "ph": "pH",
        "temperature": "তাপমাত্রা (°C)",
        "humidity": "আর্দ্রতা (%)",
        "rainfall": "বৃষ্টি (mm)",
        "manual": "🌱 অন্যান্য তথ্য দিন",
        "button": "ফসলের সুপারিশ করুন",
        "crop_title": "🌾 শীর্ষ ৩টি প্রস্তাবিত ফসল",
        "not_detected": "পাওয়া যায়নি",
        "uploaded": "আপলোড করা মাটি স্বাস্থ্য কার্ড"
    }
}


# =========================================================
# LOGO + TITLE
# =========================================================

col1, col2 = st.columns([1, 3])

with col1:

    st.image(
        "krishived_logo.jpeg",
        width=200
    )

with col2:

    st.title(
        "Krishived"
    )


# =========================================================
# LANGUAGE SELECTOR
# =========================================================

language = st.selectbox(
    "🌐 Choose Language",
    list(lang_codes.keys())
)


# =========================================================
# TRANSLATION FUNCTION
# =========================================================

def translate_text(text, lang):

    try:

        return GoogleTranslator(
            source="auto",
            target=lang
        ).translate(
            str(text)
        )

    except Exception:

        return text


# =========================================================
# OCR FUNCTION
# =========================================================

def perform_ocr(image):

    img = np.array(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.equalizeHist(
        gray
    )

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text = pytesseract.image_to_string(
        thresh,
        config="--psm 6"
    )

    return text


# =========================================================
# EXTRACT VALUES FROM SOIL HEALTH CARD
# =========================================================

def extract_soil_values(text):

    text = text.replace(",", ".")

    values = {
        "N": None,
        "P2O5": None,
        "K2O": None,
        "pH": None
    }


    # =====================================================
    # pH
    # =====================================================

    ph_patterns = [
        r'pH\s*[\(\[]?.*?[:\-]?\s*(\d+(?:\.\d+)?)',
        r'pH[^0-9]*(\d+\.\d+)',
        r'PH[^0-9]*(\d+\.\d+)'
    ]

    for pattern in ph_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            try:

                ph_value = float(
                    match.group(1)
                )

                if 2 <= ph_value <= 10:

                    values["pH"] = ph_value
                    break

            except:

                pass


    # =====================================================
    # NITROGEN
    # =====================================================

    nitrogen_patterns = [
        r'Available\s*Nitrogen.*?(\d+(?:\.\d+)?)',
        r'Nitrogen\s*\(N\).*?(\d+(?:\.\d+)?)',
        r'Nitrogen.*?(\d+(?:\.\d+)?)'
    ]

    for pattern in nitrogen_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            try:

                n = float(
                    match.group(1)
                )

                if 1 <= n <= 1000:

                    values["N"] = n
                    break

            except:

                pass


    # =====================================================
    # PHOSPHORUS - P2O5
    # =====================================================

    phosphorus_patterns = [
        r'Available\s*Phosphorus.*?(\d+(?:\.\d+)?)',
        r'Phosphorus\s*\(P.?O.?5?\).*?(\d+(?:\.\d+)?)',
        r'Phosphorus.*?(\d+(?:\.\d+)?)'
    ]

    for pattern in phosphorus_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            try:

                p = float(
                    match.group(1)
                )

                if 0 <= p <= 500:

                    values["P2O5"] = p
                    break

            except:

                pass


    # =====================================================
    # POTASSIUM - K2O
    # =====================================================

    potassium_patterns = [
        r'Available\s*Potassium.*?(\d+(?:\.\d+)?)',
        r'Potassium\s*\(K.?O.?5?\).*?(\d+(?:\.\d+)?)',
        r'Potassium.*?(\d+(?:\.\d+)?)'
    ]

    for pattern in potassium_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:

            try:

                k = float(
                    match.group(1)
                )

                if 0 <= k <= 1000:

                    values["K2O"] = k
                    break

            except:

                pass


    return values


# =========================================================
# UPLOAD KVK / SOIL HEALTH CARD
# =========================================================

st.subheader(
    labels[language]["upload"]
)


uploaded_file = st.file_uploader(

    labels[language]["upload_help"],

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# DEFAULT VALUES
# =========================================================

extracted_N = None
extracted_P2O5 = None
extracted_K2O = None

extracted_P = None
extracted_K = None

extracted_pH = None


# =========================================================
# PROCESS UPLOADED REPORT
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption=labels[language]["uploaded"],
        use_container_width=True
    )

    with st.spinner(
        "Reading Soil Health Card..."
    ):

        ocr_text = perform_ocr(
            image
        )

        soil_values = extract_soil_values(
            ocr_text
        )


    extracted_N = soil_values["N"]
    extracted_P2O5 = soil_values["P2O5"]
    extracted_K2O = soil_values["K2O"]
    extracted_pH = soil_values["pH"]


    # =====================================================
    # CONVERT P2O5 → P
    # =====================================================

    if extracted_P2O5 is not None:

        extracted_P = (
            extracted_P2O5 * 0.4364
        )


    # =====================================================
    # CONVERT K2O → K
    # =====================================================

    if extracted_K2O is not None:

        extracted_K = (
            extracted_K2O * 0.8301
        )


    # =====================================================
    # SHOW EXTRACTED INFORMATION
    # =====================================================

    st.subheader(
        labels[language]["extracted"]
    )

    col1, col2 = st.columns(2)


    with col1:

        if extracted_N is not None:

            st.write(
                f"**Nitrogen (N):** "
                f"{extracted_N}"
            )

        else:

            st.write(
                f"**Nitrogen (N):** "
                f"{labels[language]['not_detected']}"
            )


        if extracted_P2O5 is not None:

            st.write(
                f"**Phosphorus (P₂O₅):** "
                f"{extracted_P2O5}"
            )

        else:

            st.write(
                f"**Phosphorus (P₂O₅):** "
                f"{labels[language]['not_detected']}"
            )


    with col2:

        if extracted_K2O is not None:

            st.write(
                f"**Potassium (K₂O):** "
                f"{extracted_K2O}"
            )

        else:

            st.write(
                f"**Potassium (K₂O):** "
                f"{labels[language]['not_detected']}"
            )


        if extracted_pH is not None:

            st.write(
                f"**pH:** "
                f"{extracted_pH}"
            )

        else:

            st.write(
                f"**pH:** "
                f"{labels[language]['not_detected']}"
            )


# =========================================================
# SOIL INFORMATION
# =========================================================

st.subheader(
    "🌱 Soil Information"
)


N = st.number_input(
    labels[language]["nitrogen"],
    min_value=0.0,
    max_value=1000.0,
    value=(
        float(extracted_N)
        if extracted_N is not None
        else 90.0
    )
)


P = st.number_input(
    labels[language]["phosphorus"],
    min_value=0.0,
    max_value=500.0,
    value=(
        float(extracted_P)
        if extracted_P is not None
        else 40.0
    )
)


K = st.number_input(
    labels[language]["potassium"],
    min_value=0.0,
    max_value=1000.0,
    value=(
        float(extracted_K)
        if extracted_K is not None
        else 40.0
    )
)


ph = st.number_input(
    labels[language]["ph"],
    min_value=0.0,
    max_value=14.0,
    value=(
        float(extracted_pH)
        if extracted_pH is not None
        else 6.5
    )
)


# =========================================================
# OTHER FARMER INPUTS
# =========================================================

st.subheader(
    labels[language]["manual"]
)


temperature = st.number_input(
    labels[language]["temperature"],
    min_value=0.0,
    max_value=50.0,
    value=25.0
)


humidity = st.number_input(
    labels[language]["humidity"],
    min_value=0.0,
    max_value=100.0,
    value=80.0
)


rainfall = st.number_input(
    labels[language]["rainfall"],
    min_value=0.0,
    max_value=5000.0,
    value=200.0
)


# =========================================================
# LOAD CROP DATASET
# =========================================================

try:

    data = pd.read_csv(
        "Crop_recommendation.csv",
        encoding="latin1"
    )

except FileNotFoundError:

    st.error(
        "Crop_recommendation.csv was not found. "
        "Please put it in the same folder as crop_chatbot.py."
    )

    st.stop()


# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_crop_columns = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]


missing_crop_columns = [
    column
    for column in required_crop_columns
    if column not in data.columns
]


if missing_crop_columns:

    st.error(
        f"Crop dataset is missing columns: "
        f"{missing_crop_columns}"
    )

    st.stop()


# =========================================================
# TRAIN CROP MODEL
# =========================================================

X = data[
    required_crop_columns
]

y = data[
    "label"
]


model = DecisionTreeClassifier(
    random_state=42
)


model.fit(
    X,
    y
)


# =========================================================
# PREDICT TOP 3 CROPS
# =========================================================

if st.button(
    labels[language]["button"],
    type="primary"
):

    crop_input = pd.DataFrame(
        [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]],
        columns=required_crop_columns
    )


    probabilities = model.predict_proba(
        crop_input
    )[0]


    crop_names = model.classes_


    top_indices = np.argsort(
        probabilities
    )[::-1]


    top_indices = top_indices[:3]


    st.subheader(
        labels[language]["crop_title"]
    )


    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        crop_name = crop_names[index]


        probability = (
            probabilities[index] * 100
        )


        translated_crop = translate_text(
            crop_name,
            lang_codes[language]
        )


        if rank == 1:

            st.success(
                f"🥇 {translated_crop}"
            )

            st.write(
                f"Model probability: "
                f"{probability:.1f}%"
            )


        elif rank == 2:

            st.info(
                f"🥈 {translated_crop}"
            )

            st.write(
                f"Model probability: "
                f"{probability:.1f}%"
            )


        elif rank == 3:

            st.warning(
                f"🥉 {translated_crop}"
            )

            st.write(
                f"Model probability: "
                f"{probability:.1f}%"
            )