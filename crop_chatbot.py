import pandas as pd
import streamlit as st
import pytesseract
import cv2
import re
import numpy as np

from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
        "choose_language": "🌐 Choose Language",

        "upload":
            "📎 Upload KVK (Krishi Vigyan Kendra) / Soil Health Card",

        "upload_help":
            "Upload a clear image of your Soil Health Card",

        "reading":
            "Reading Soil Health Card...",

        "extracted":
            "📄 Information extracted from your report",

        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "ph": "pH",

        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "rainfall": "Rainfall (mm)",

        "manual": "🌱 Enter other information",
        "soil_information": "🌱 Soil Information",

        "button": "Recommend Crops",

        "crop_title": "🌾 Top Crops to Grow",

        "not_detected": "Not detected",

        "uploaded": "Uploaded Soil Health Card",

        "probability": "Model probability",

        "no_crop":
            "No crop has a probability of at least 10% for these conditions.",

        "dataset_error":
            "Crop_recommendation.csv was not found. Please put it in the same folder as crop_chatbot.py.",

        "missing_columns":
            "Crop dataset is missing columns:",

        "nitrogen_extracted": "Nitrogen (N)",
        "phosphorus_extracted": "Phosphorus (P)",
        "potassium_extracted": "Potassium (K)"
    },

    "Hindi": {
        "title": "कृषिवेद",
        "choose_language": "🌐 भाषा चुनें",

        "upload":
            "📎 KVK (कृषि विज्ञान केंद्र) / मृदा स्वास्थ्य कार्ड अपलोड करें",

        "upload_help":
            "अपने मृदा स्वास्थ्य कार्ड की स्पष्ट तस्वीर अपलोड करें",

        "reading":
            "मृदा स्वास्थ्य कार्ड पढ़ा जा रहा है...",

        "extracted":
            "📄 आपकी रिपोर्ट से प्राप्त जानकारी",

        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फॉस्फोरस (P)",
        "potassium": "पोटैशियम (K)",
        "ph": "pH",

        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "rainfall": "वर्षा (mm)",

        "manual": "🌱 अन्य जानकारी दर्ज करें",
        "soil_information": "🌱 मिट्टी की जानकारी",

        "button": "फसल सुझाएं",

        "crop_title": "🌾 उगाने के लिए शीर्ष फसलें",

        "not_detected": "पता नहीं चला",

        "uploaded": "अपलोड किया गया मृदा स्वास्थ्य कार्ड",

        "probability": "मॉडल संभावना",

        "no_crop":
            "इन परिस्थितियों के लिए किसी भी फसल की संभावना कम से कम 10% नहीं है।",

        "dataset_error":
            "Crop_recommendation.csv नहीं मिली। कृपया इसे crop_chatbot.py के साथ उसी फ़ोल्डर में रखें।",

        "missing_columns":
            "क्रॉप डेटासेट में निम्न कॉलम नहीं हैं:",

        "nitrogen_extracted": "नाइट्रोजन (N)",
        "phosphorus_extracted": "फॉस्फोरस (P)",
        "potassium_extracted": "पोटैशियम (K)"
    },

    "Kannada": {
        "title": "ಕೃಷಿವೇದ",
        "choose_language": "🌐 ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",

        "upload":
            "📎 KVK (ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ) / ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಅಪ್ಲೋಡ್ ಮಾಡಿ",

        "upload_help":
            "ನಿಮ್ಮ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್‌ನ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",

        "reading":
            "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಓದಲಾಗುತ್ತಿದೆ...",

        "extracted":
            "📄 ನಿಮ್ಮ ವರದಿಯಿಂದ ಪಡೆದ ಮಾಹಿತಿ",

        "nitrogen": "ನೈಟ್ರಜನ (N)",
        "phosphorus": "ಫಾಸ್ಫರಸ್ (P)",
        "potassium": "ಪೊಟ್ಯಾಸಿಯಮ್ (K)",
        "ph": "pH",

        "temperature": "ತಾಪಮಾನ (°C)",
        "humidity": "ಆರ್ದ್ರತೆ (%)",
        "rainfall": "ಮಳೆ (mm)",

        "manual": "🌱 ಇತರ ಮಾಹಿತಿಯನ್ನು ನಮೂದಿಸಿ",
        "soil_information": "🌱 ಮಣ್ಣಿನ ಮಾಹಿತಿ",

        "button": "ಬೆಳೆಗಳನ್ನು ಸೂಚಿಸಿ",

        "crop_title": "🌾 ಬೆಳೆಯಲು ಸೂಕ್ತವಾದ ಪ್ರಮುಖ ಬೆಳೆಗಳು",

        "not_detected": "ಪತ್ತೆಯಾಗಿಲ್ಲ",

        "uploaded": "ಅಪ್ಲೋಡ್ ಮಾಡಿದ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್",

        "probability": "ಮಾದರಿ ಸಂಭವನೀಯತೆ",

        "no_crop":
            "ಈ ಪರಿಸ್ಥಿತಿಗಳಿಗೆ ಕನಿಷ್ಠ 10% ಸಂಭವನೀಯತೆಯ ಬೆಳೆ ಇಲ್ಲ.",

        "dataset_error":
            "Crop_recommendation.csv ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಅದನ್ನು crop_chatbot.py ಇರುವ ಅದೇ ಫೋಲ್ಡರ್‌ನಲ್ಲಿ ಇರಿಸಿ.",

        "missing_columns":
            "ಕ್ರಾಪ್ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಈ ಕಾಲಮ್‌ಗಳು ಕಾಣೆಯಾಗಿವೆ:",

        "nitrogen_extracted": "ನೈಟ್ರಜನ (N)",
        "phosphorus_extracted": "ಫಾಸ್ಫರಸ್ (P)",
        "potassium_extracted": "ಪೊಟ್ಯಾಸಿಯಮ್ (K)"
    },

    "Punjabi": {
        "title": "ਕ੍ਰਿਸ਼ਿਵੇਦ",
        "choose_language": "🌐 ਭਾਸ਼ਾ ਚੁਣੋ",

        "upload":
            "📎 KVK (ਕ੍ਰਿਸ਼ੀ ਵਿਗਿਆਨ ਕੇਂਦਰ) / ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਅਪਲੋਡ ਕਰੋ",

        "upload_help":
            "ਆਪਣੇ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਦੀ ਸਾਫ਼ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",

        "reading":
            "ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਪੜ੍ਹਿਆ ਜਾ ਰਿਹਾ ਹੈ...",

        "extracted":
            "📄 ਤੁਹਾਡੀ ਰਿਪੋਰਟ ਤੋਂ ਪ੍ਰਾਪਤ ਜਾਣਕਾਰੀ",

        "nitrogen": "ਨਾਈਟ੍ਰੋਜਨ (N)",
        "phosphorus": "ਫਾਸਫੋਰਸ (P)",
        "potassium": "ਪੋਟਾਸ਼ੀਅਮ (K)",
        "ph": "pH",

        "temperature": "ਤਾਪਮਾਨ (°C)",
        "humidity": "ਨਮੀ (%)",
        "rainfall": "ਵਰਖਾ (mm)",

        "manual": "🌱 ਹੋਰ ਜਾਣਕਾਰੀ ਦਰਜ ਕਰੋ",
        "soil_information": "🌱 ਮਿੱਟੀ ਦੀ ਜਾਣਕਾਰੀ",

        "button": "ਫਸਲਾਂ ਦੀ ਸਿਫਾਰਸ਼ ਕਰੋ",

        "crop_title": "🌾 ਉਗਾਉਣ ਲਈ ਮੁੱਖ ਫਸਲਾਂ",

        "not_detected": "ਪਤਾ ਨਹੀਂ ਲੱਗਿਆ",

        "uploaded": "ਅਪਲੋਡ ਕੀਤਾ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ",

        "probability": "ਮਾਡਲ ਸੰਭਾਵਨਾ",

        "no_crop":
            "ਇਨ੍ਹਾਂ ਹਾਲਾਤਾਂ ਲਈ ਕਿਸੇ ਵੀ ਫਸਲ ਦੀ ਸੰਭਾਵਨਾ ਘੱਟੋ-ਘੱਟ 10% ਨਹੀਂ ਹੈ।",

        "dataset_error":
            "Crop_recommendation.csv ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ crop_chatbot.py ਵਾਲੇ ਫੋਲਡਰ ਵਿੱਚ ਰੱਖੋ।",

        "missing_columns":
            "ਕ੍ਰਾਪ ਡੇਟਾਸੈੱਟ ਵਿੱਚ ਇਹ ਕਾਲਮ ਮੌਜੂਦ ਨਹੀਂ ਹਨ:",

        "nitrogen_extracted": "ਨਾਈਟ੍ਰੋਜਨ (N)",
        "phosphorus_extracted": "ਫਾਸਫੋਰਸ (P)",
        "potassium_extracted": "ਪੋਟਾਸ਼ੀਅਮ (K)"
    },

    "Bengali": {
        "title": "কৃষিবেদ",
        "choose_language": "🌐 ভাষা নির্বাচন করুন",

        "upload":
            "📎 KVK (কৃষি বিজ্ঞান কেন্দ্র) / মাটি স্বাস্থ্য কার্ড আপলোড করুন",

        "upload_help":
            "আপনার মাটি স্বাস্থ্য কার্ডের একটি পরিষ্কার ছবি আপলোড করুন",

        "reading":
            "মাটি স্বাস্থ্য কার্ড পড়া হচ্ছে...",

        "extracted":
            "📄 আপনার রিপোর্ট থেকে প্রাপ্ত তথ্য",

        "nitrogen": "নাইট্রোজেন (N)",
        "phosphorus": "ফসফরাস (P)",
        "potassium": "পটাশিয়াম (K)",
        "ph": "pH",

        "temperature": "তাপমাত্রা (°C)",
        "humidity": "আর্দ্রতা (%)",
        "rainfall": "বৃষ্টি (mm)",

        "manual": "🌱 অন্যান্য তথ্য দিন",
        "soil_information": "🌱 মাটির তথ্য",

        "button": "ফসলের সুপারিশ করুন",

        "crop_title": "🌾 চাষের জন্য প্রধান ফসল",

        "not_detected": "পাওয়া যায়নি",

        "uploaded": "আপলোড করা মাটি স্বাস্থ্য কার্ড",

        "probability": "মডেল সম্ভাবনা",

        "no_crop":
            "এই পরিস্থিতিতে কোনো ফসলের সম্ভাবনা কমপক্ষে 10% নয়।",

        "dataset_error":
            "Crop_recommendation.csv পাওয়া যায়নি। অনুগ্রহ করে এটি crop_chatbot.py-এর একই ফোল্ডারে রাখুন।",

        "missing_columns":
            "ক্রপ ডেটাসেটে নিম্নলিখিত কলামগুলি নেই:",

        "nitrogen_extracted": "নাইট্রোজেন (N)",
        "phosphorus_extracted": "ফসফরাস (P)",
        "potassium_extracted": "পটাশিয়াম (K)"
    }
}


# =========================================================
# CROP TRANSLATIONS
# =========================================================

crop_translations = {

    "rice": {
        "en": "Rice",
        "hi": "चावल",
        "kn": "ಅಕ್ಕಿ",
        "pa": "ਚੌਲ",
        "bn": "ধান"
    },

    "maize": {
        "en": "Maize",
        "hi": "मक्का",
        "kn": "ಮೆಕ್ಕೆಜೋಳ",
        "pa": "ਮੱਕੀ",
        "bn": "ভুট্টা"
    },

    "chickpea": {
        "en": "Chickpea",
        "hi": "चना",
        "kn": "ಕಡಲೆ",
        "pa": "ਛੋਲੇ",
        "bn": "ছোলা"
    },

    "kidneybeans": {
        "en": "Kidney Beans",
        "hi": "राजमा",
        "kn": "ರಾಜ್ಮಾ",
        "pa": "ਰਾਜਮਾਂਹ",
        "bn": "রাজমা"
    },

    "pigeonpeas": {
        "en": "Pigeon Peas",
        "hi": "अरहर",
        "kn": "ತೊಗರಿ ಬೇಳೆ",
        "pa": "ਅਰਹਰ",
        "bn": "অড়হর"
    },

    "mothbeans": {
        "en": "Moth Beans",
        "hi": "मोठ",
        "kn": "ಮೋತ್ ಬೀನ್ಸ್",
        "pa": "ਮੋਠ",
        "bn": "মট"
    },

    "mungbean": {
        "en": "Mung Bean",
        "hi": "मूंग",
        "kn": "ಹೆಸರು ಕಾಳು",
        "pa": "ਮੂੰਗ",
        "bn": "মুগ ডাল"
    },

    "blackgram": {
        "en": "Black Gram",
        "hi": "उड़द",
        "kn": "ಉದ್ದು",
        "pa": "ਉੜਦ",
        "bn": "বিউলি ডাল"
    },

    "lentil": {
        "en": "Lentil",
        "hi": "मसूर",
        "kn": "ಮಸೂರ್ ಬೇಳೆ",
        "pa": "ਮਸੂਰ",
        "bn": "মসুর ডাল"
    },

    "pomegranate": {
        "en": "Pomegranate",
        "hi": "अनार",
        "kn": "ದಾಳಿಂಬೆ",
        "pa": "ਅਨਾਰ",
        "bn": "ডালিম"
    },

    "banana": {
        "en": "Banana",
        "hi": "केला",
        "kn": "ಬಾಳೆಹಣ್ಣು",
        "pa": "ਕੇਲਾ",
        "bn": "কলা"
    },

    "mango": {
        "en": "Mango",
        "hi": "आम",
        "kn": "ಮಾವು",
        "pa": "ਅੰਬ",
        "bn": "আম"
    },

    "grapes": {
        "en": "Grapes",
        "hi": "अंगूर",
        "kn": "ದ್ರಾಕ್ಷಿ",
        "pa": "ਅੰਗੂਰ",
        "bn": "আঙুর"
    },

    "watermelon": {
        "en": "Watermelon",
        "hi": "तरबूज",
        "kn": "ಕಲ್ಲಂಗಡಿ",
        "pa": "ਤਰਬੂਜ",
        "bn": "তরমুজ"
    },

    "muskmelon": {
        "en": "Muskmelon",
        "hi": "खरबूजा",
        "kn": "ಖರ್ಬೂಜ",
        "pa": "ਖਰਬੂਜਾ",
        "bn": "খরমুজ"
    },

    "apple": {
        "en": "Apple",
        "hi": "सेब",
        "kn": "ಸೇಬು",
        "pa": "ਸੇਬ",
        "bn": "আপেল"
    },

    "orange": {
        "en": "Orange",
        "hi": "संतरा",
        "kn": "ಕಿತ್ತಳೆ",
        "pa": "ਸੰਤਰਾ",
        "bn": "কমলা"
    },

    "papaya": {
        "en": "Papaya",
        "hi": "पपीता",
        "kn": "ಪಪ್ಪಾಯಿ",
        "pa": "ਪਪੀਤਾ",
        "bn": "পেঁপে"
    },

    "coconut": {
        "en": "Coconut",
        "hi": "नारियल",
        "kn": "ತೆಂಗಿನಕಾಯಿ",
        "pa": "ਨਾਰੀਅਲ",
        "bn": "নারকেল"
    },

    "cotton": {
        "en": "Cotton",
        "hi": "कपास",
        "kn": "ಹತ್ತಿ",
        "pa": "ਕਪਾਹ",
        "bn": "তুলা"
    },

    "jute": {
        "en": "Jute",
        "hi": "जूट",
        "kn": "ಸೆಣಬು",
        "pa": "ਜੂਟ",
        "bn": "পাট"
    },

    "coffee": {
        "en": "Coffee",
        "hi": "कॉफी",
        "kn": "ಕಾಫಿ",
        "pa": "ਕੌਫੀ",
        "bn": "কফি"
    }
}


# =========================================================
# CROP TRANSLATION
# =========================================================

def translate_crop_name(crop_name, language_code):

    crop_name = str(crop_name).strip().lower()

    if crop_name in crop_translations:

        return crop_translations[crop_name].get(
            language_code,
            crop_name
        )

    return crop_name


# =========================================================
# LANGUAGE SELECTOR
# =========================================================

language = st.selectbox(
    labels["English"]["choose_language"],
    list(lang_codes.keys())
)


# =========================================================
# LOGO + TITLE
# =========================================================

col1, col2 = st.columns([1, 3])

with col1:

    try:

        st.image(
            "krishived_logo.jpeg",
            width=200
        )

    except Exception:

        st.write("🌾")


with col2:

    st.title(
        labels[language]["title"]
    )


# =========================================================
# OCR PREPROCESSING
# =========================================================

def preprocess_for_ocr(image):

    img = np.array(image)

    if len(img.shape) == 3:

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2GRAY
        )

    else:

        gray = img

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        gray,
        enhanced,
        otsu,
        adaptive
    ]


# =========================================================
# OCR
# =========================================================

def run_multiple_ocr(image):

    versions = preprocess_for_ocr(image)

    all_text = []

    best_data = None
    best_confidence = -1

    for img in versions:

        for psm in [6, 11, 12]:

            text = pytesseract.image_to_string(
                img,
                config=f"--psm {psm}"
            )

            if text:

                all_text.append(text)

            data = pytesseract.image_to_data(
                img,
                config=f"--psm {psm}",
                output_type=pytesseract.Output.DICT
            )

            confidences = []

            for conf in data["conf"]:

                try:

                    conf = float(conf)

                    if conf > 0:

                        confidences.append(conf)

                except Exception:

                    pass

            if confidences:

                avg_confidence = np.mean(
                    confidences
                )

                if avg_confidence > best_confidence:

                    best_confidence = avg_confidence
                    best_data = data

    combined_text = "\n".join(all_text)

    return combined_text, best_data


# =========================================================
# WORD POSITIONS
# =========================================================

def get_word_positions(ocr_data):

    words = []

    if ocr_data is None:

        return words

    count = len(
        ocr_data["text"]
    )

    # OCR images are enlarged 3x.
    # Convert coordinates back to original image size.

    OCR_SCALE = 3.0

    for i in range(count):

        text = str(
            ocr_data["text"][i]
        ).strip()

        if not text:

            continue

        try:

            confidence = float(
                ocr_data["conf"][i]
            )

        except Exception:

            confidence = 0

        words.append({

            "text": text,

            "x": int(
                ocr_data["left"][i] / OCR_SCALE
            ),

            "y": int(
                ocr_data["top"][i] / OCR_SCALE
            ),

            "w": int(
                ocr_data["width"][i] / OCR_SCALE
            ),

            "h": int(
                ocr_data["height"][i] / OCR_SCALE
            ),

            "conf": confidence
        })

    return words


# =========================================================
# NUMERIC OCR WORDS
# =========================================================

def get_numeric_words(ocr_data):

    numeric_words = []

    words = get_word_positions(
        ocr_data
    )

    for word in words:

        text = word["text"]

        fixed = (
            text
            .replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
            .replace("|", "1")
            .replace(",", ".")
        )

        match = re.search(
            r'\d+(?:\.\d+)?',
            fixed
        )

        if match:

            try:

                value = float(
                    match.group()
                )

                numeric_words.append({

                    "value": value,

                    "x": word["x"],

                    "y": word["y"],

                    "w": word["w"],

                    "h": word["h"],

                    "text": text
                })

            except Exception:

                pass

    return numeric_words


# =========================================================
# EXTRACT N, P, K FROM FIRST ROW
# =========================================================

def extract_npk_from_first_row(
    image,
    ocr_data
):

    numbers = get_numeric_words(
        ocr_data
    )

    if not numbers:

        return {
            "N": None,
            "P": None,
            "K": None
        }

    candidates = []

    for number in numbers:

        raw_text = str(
            number["text"]
        ).strip()

        fixed = (
            raw_text
            .replace(",", ".")
            .replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("l", "1")
            .replace("|", "1")
        )

        # Soil Health Card main N/P/K values
        # are normally written with decimal values.
        decimal_match = re.fullmatch(
            r'\d+\.\d{1,3}',
            fixed
        )

        if not decimal_match:

            continue

        try:

            value = float(fixed)

        except Exception:

            continue

        if not (
            0 <= value <= 1000
        ):

            continue

        candidates.append({

            "value": value,

            "x": number["x"],

            "y": number["y"],

            "text": raw_text

        })

    if not candidates:

        return {
            "N": None,
            "P": None,
            "K": None
        }

    # -----------------------------------------------------
    # GROUP DECIMAL VALUES BY ROW
    # -----------------------------------------------------

    candidates.sort(
        key=lambda item: item["y"]
    )

    rows = []

    ROW_TOLERANCE = 60

    for candidate in candidates:

        added = False

        for row in rows:

            average_y = np.mean([
                item["y"]
                for item in row
            ])

            if abs(
                candidate["y"] - average_y
            ) <= ROW_TOLERANCE:

                row.append(candidate)

                added = True

                break

        if not added:

            rows.append(
                [candidate]
            )

    # -----------------------------------------------------
    # FIND FIRST ROW WITH AT LEAST 3 VALUES
    # -----------------------------------------------------

    valid_rows = [

        row

        for row in rows

        if len(row) >= 3

    ]

    if valid_rows:

        # The first valid 3-value row is the N/P/K row.

        npk_row = sorted(
            valid_rows,
            key=lambda row: min(
                item["y"]
                for item in row
            )
        )[0]

        # Left → Right:
        # Nitrogen → Phosphorus → Potassium

        npk_row = sorted(
            npk_row,
            key=lambda item: item["x"]
        )

        return {

            "N": npk_row[0]["value"],

            "P": npk_row[1]["value"],

            "K": npk_row[2]["value"]

        }

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    # If OCR only detects two or fewer decimal values,
    # do not shift values between nutrients.

    return {
        "N": None,
        "P": None,
        "K": None
    }


# =========================================================
# pH DETECTION
# =========================================================

def detect_ph_from_text(text):

    if not text:

        return None

    patterns = [

        r'p\s*H\s*[:\-]?\s*(\d+(?:\.\d+)?)',

        r'ph\s*[:\-]?\s*(\d+(?:\.\d+)?)',

        r'PH\s*[:\-]?\s*(\d+(?:\.\d+)?)'

    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        for match in matches:

            try:

                value = float(match)

                if 3.5 <= value <= 9.9:

                    return value

            except Exception:

                pass

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    candidates = re.findall(
        r'\b\d+\.\d+\b',
        text
    )

    for candidate in candidates:

        try:

            value = float(candidate)

            if 3.5 <= value <= 9.9:

                return value

        except Exception:

            pass

    return None


# =========================================================
# EXTRACT SOIL VALUES
# =========================================================

def extract_soil_values(image):

    values = {

        "N": None,
        "P": None,
        "K": None,
        "pH": None

    }

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    text, ocr_data = run_multiple_ocr(
        image
    )

    # -----------------------------------------------------
    # pH
    # -----------------------------------------------------

    values["pH"] = detect_ph_from_text(
        text
    )

    # -----------------------------------------------------
    # N, P, K
    # -----------------------------------------------------

    npk_values = extract_npk_from_first_row(
        image,
        ocr_data
    )

    values["N"] = npk_values["N"]
    values["P"] = npk_values["P"]
    values["K"] = npk_values["K"]

    return values


# =========================================================
# UPLOAD
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
extracted_P = None
extracted_K = None
extracted_pH = None


# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # -----------------------------------------------------
    # SHOW IMAGE
    # -----------------------------------------------------

    st.image(

        image,

        caption=labels[language]["uploaded"],

        use_container_width=True

    )

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------

    with st.spinner(
        labels[language]["reading"]
    ):

        soil_values = extract_soil_values(
            image
        )

    # -----------------------------------------------------
    # VALUES
    # -----------------------------------------------------

    extracted_N = soil_values["N"]
    extracted_P = soil_values["P"]
    extracted_K = soil_values["K"]
    extracted_pH = soil_values["pH"]

    # -----------------------------------------------------
    # DISPLAY EXTRACTED
    # -----------------------------------------------------

    st.subheader(
        labels[language]["extracted"]
    )

    col1, col2 = st.columns(2)

    with col1:

        if extracted_N is not None:

            st.write(

                f"**{labels[language]['nitrogen_extracted']}:** "
                f"{extracted_N:.2f} kg/ha"

            )

        else:

            st.write(

                f"**{labels[language]['nitrogen_extracted']}:** "
                f"{labels[language]['not_detected']}"

            )

        if extracted_P is not None:

            st.write(

                f"**{labels[language]['phosphorus_extracted']}:** "
                f"{extracted_P:.2f} kg/ha"

            )

        else:

            st.write(

                f"**{labels[language]['phosphorus_extracted']}:** "
                f"{labels[language]['not_detected']}"

            )

    with col2:

        if extracted_K is not None:

            st.write(

                f"**{labels[language]['potassium_extracted']}:** "
                f"{extracted_K:.2f} kg/ha"

            )

        else:

            st.write(

                f"**{labels[language]['potassium_extracted']}:** "
                f"{labels[language]['not_detected']}"

            )

        if extracted_pH is not None:

            st.write(

                f"**{labels[language]['ph']}:** "
                f"{extracted_pH:.2f}"

            )

        else:

            st.write(

                f"**{labels[language]['ph']}:** "
                f"{labels[language]['not_detected']}"

            )


# =========================================================
# SOIL INFORMATION
# =========================================================

st.subheader(
    labels[language]["soil_information"]
)


# =========================================================
# N
# =========================================================

N = st.number_input(

    labels[language]["nitrogen"],

    min_value=0.0,

    max_value=1000.0,

    value=(

        float(extracted_N)

        if extracted_N is not None

        else 90.0

    ),

    step=1.0

)


# =========================================================
# P
# =========================================================

P = st.number_input(

    labels[language]["phosphorus"],

    min_value=0.0,

    max_value=500.0,

    value=(

        float(extracted_P)

        if extracted_P is not None

        else 40.0

    ),

    step=1.0

)


# =========================================================
# K
# =========================================================

K = st.number_input(

    labels[language]["potassium"],

    min_value=0.0,

    max_value=1000.0,

    value=(

        float(extracted_K)

        if extracted_K is not None

        else 40.0

    ),

    step=1.0

)


# =========================================================
# pH
# =========================================================

ph = st.number_input(

    labels[language]["ph"],

    min_value=0.0,

    max_value=14.0,

    value=(

        float(extracted_pH)

        if extracted_pH is not None

        else 6.5

    ),

    step=0.1

)


# =========================================================
# OTHER INFORMATION
# =========================================================

st.subheader(
    labels[language]["manual"]
)


temperature = st.number_input(

    labels[language]["temperature"],

    min_value=0.0,

    max_value=50.0,

    value=25.0,

    step=0.1

)


humidity = st.number_input(

    labels[language]["humidity"],

    min_value=0.0,

    max_value=100.0,

    value=80.0,

    step=1.0

)


rainfall = st.number_input(

    labels[language]["rainfall"],

    min_value=0.0,

    max_value=5000.0,

    value=200.0,

    step=1.0

)


# =========================================================
# LOAD DATASET
# =========================================================

try:

    data = pd.read_csv(
        "Crop_recommendation.csv",
        encoding="latin1"
    )

except FileNotFoundError:

    st.error(
        labels[language]["dataset_error"]
    )

    st.stop()


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

data.columns = (

    data.columns
    .astype(str)
    .str.strip()

)


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

        f"{labels[language]['missing_columns']} "
        f"{missing_crop_columns}"

    )

    st.stop()


# =========================================================
# LABEL COLUMN
# =========================================================

if "label" not in data.columns:

    st.error(
        "Crop dataset must contain a 'label' column."
    )

    st.stop()


# =========================================================
# NUMERIC CONVERSION
# =========================================================

for column in required_crop_columns:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )


data = data.dropna(
    subset=required_crop_columns + ["label"]
)


# =========================================================
# X AND Y
# =========================================================

X = data[
    required_crop_columns
]

y = data[
    "label"
]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=42,

    stratify=y

)


# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestClassifier(

    n_estimators=100,

    random_state=42

)


model.fit(
    X_train,
    y_train
)


# =========================================================
# CROP PREDICTION
# =========================================================

if st.button(

    labels[language]["button"],

    type="primary"

):

    # -----------------------------------------------------
    # NITROGEN CALIBRATION
    # -----------------------------------------------------

    N_for_model = N * 0.30

    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    crop_input = pd.DataFrame(

        [[

            N_for_model,

            P,

            K,

            temperature,

            humidity,

            ph,

            rainfall

        ]],

        columns=required_crop_columns

    )

    # -----------------------------------------------------
    # PREDICT PROBABILITIES
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        crop_input
    )[0]

    crop_names = model.classes_

    # -----------------------------------------------------
    # SORT PROBABILITIES
    # -----------------------------------------------------

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    # -----------------------------------------------------
    # CROPS >= 10%
    # -----------------------------------------------------

    valid_indices = []

    for index in sorted_indices:

        probability_percent = (

            probabilities[index]
            * 100

        )

        if probability_percent >= 10.0:

            valid_indices.append(
                index
            )

    # -----------------------------------------------------
    # TOP 3
    # -----------------------------------------------------

    top_indices = valid_indices[:3]

    # -----------------------------------------------------
    # RESULTS HEADING
    # -----------------------------------------------------

    st.subheader(
        labels[language]["crop_title"]
    )

    # -----------------------------------------------------
    # NO RESULTS
    # -----------------------------------------------------

    if len(top_indices) == 0:

        st.warning(
            labels[language]["no_crop"]
        )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        crop_name = crop_names[index]

        probability = (

            probabilities[index]
            * 100

        )

        translated_crop = translate_crop_name(

            crop_name,

            lang_codes[language]

        )

        if rank == 1:

            st.success(
                f"🥇 {translated_crop}"
            )

        elif rank == 2:

            st.info(
                f"🥈 {translated_crop}"
            )

        elif rank == 3:

            st.warning(
                f"🥉 {translated_crop}"
            )

        st.write(

            f"{labels[language]['probability']}: "
            f"{probability:.2f}%"

        )