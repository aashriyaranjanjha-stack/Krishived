import pandas as pd
import streamlit as st
import pytesseract
import cv2
import re
import numpy as np

from PIL import Image
from sklearn.ensemble import RandomForestClassifier


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

    # =====================================================
    # ENGLISH
    # =====================================================

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

        "nitrogen":
            "Nitrogen (N)",

        "phosphorus":
            "Phosphorus (P)",

        "potassium":
            "Potassium (K)",

        "ph":
            "pH",

        "p2o5":
            "Phosphorus (P₂O₅)",

        "k2o":
            "Potassium (K₂O)",

        "temperature":
            "Temperature (°C)",

        "humidity":
            "Humidity (%)",

        "rainfall":
            "Rainfall (mm)",

        "manual":
            "🌱 Enter other information",

        "soil_information":
            "🌱 Soil Information",

        "button":
            "Recommend Crops",

        "crop_title":
            "🌾 Top Crops to Grow",

        "not_detected":
            "Not detected",

        "uploaded":
            "Uploaded Soil Health Card",

        "probability":
            "Model probability",

        "no_crop":
            "No crop has a probability of at least 10% for these conditions.",

        "dataset_error":
            "Crop_recommendation.csv was not found. Please put it in the same folder as crop_chatbot.py.",

        "missing_columns":
            "Crop dataset is missing columns:",

        "nitrogen_extracted":
            "Nitrogen (N)",

        "phosphorus_extracted":
            "Phosphorus (P₂O₅)",

        "potassium_extracted":
            "Potassium (K₂O)"
    },


    # =====================================================
    # HINDI
    # =====================================================

    "Hindi": {

        "title":
            "कृषिवेद",

        "choose_language":
            "🌐 भाषा चुनें",

        "upload":
            "📎 KVK (कृषि विज्ञान केंद्र) / मृदा स्वास्थ्य कार्ड अपलोड करें",

        "upload_help":
            "अपने मृदा स्वास्थ्य कार्ड की स्पष्ट तस्वीर अपलोड करें",

        "reading":
            "मृदा स्वास्थ्य कार्ड पढ़ा जा रहा है...",

        "extracted":
            "📄 आपकी रिपोर्ट से प्राप्त जानकारी",

        "nitrogen":
            "नाइट्रोजन (N)",

        "phosphorus":
            "फॉस्फोरस (P)",

        "potassium":
            "पोटैशियम (K)",

        "ph":
            "pH",

        "p2o5":
            "फॉस्फोरस (P₂O₅)",

        "k2o":
            "पोटैशियम (K₂O)",

        "temperature":
            "तापमान (°C)",

        "humidity":
            "आर्द्रता (%)",

        "rainfall":
            "वर्षा (mm)",

        "manual":
            "🌱 अन्य जानकारी दर्ज करें",

        "soil_information":
            "🌱 मिट्टी की जानकारी",

        "button":
            "फसल सुझाएं",

        "crop_title":
            "🌾 उगाने के लिए शीर्ष फसलें",

        "not_detected":
            "पता नहीं चला",

        "uploaded":
            "अपलोड किया गया मृदा स्वास्थ्य कार्ड",

        "probability":
            "मॉडल संभावना",

        "no_crop":
            "इन परिस्थितियों के लिए किसी भी फसल की संभावना कम से कम 10% नहीं है।",

        "dataset_error":
            "Crop_recommendation.csv नहीं मिली। कृपया इसे crop_chatbot.py के साथ उसी फ़ोल्डर में रखें।",

        "missing_columns":
            "क्रॉप डेटासेट में निम्न कॉलम नहीं हैं:",

        "nitrogen_extracted":
            "नाइट्रोजन (N)",

        "phosphorus_extracted":
            "फॉस्फोरस (P₂O₅)",

        "potassium_extracted":
            "पोटैशियम (K₂O)"
    },


    # =====================================================
    # KANNADA
    # =====================================================

    "Kannada": {

        "title":
            "ಕೃಷಿವೇದ",

        "choose_language":
            "🌐 ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",

        "upload":
            "📎 KVK (ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ) / ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಅಪ್ಲೋಡ್ ಮಾಡಿ",

        "upload_help":
            "ನಿಮ್ಮ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್‌ನ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ",

        "reading":
            "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಓದಲಾಗುತ್ತಿದೆ...",

        "extracted":
            "📄 ನಿಮ್ಮ ವರದಿಯಿಂದ ಪಡೆದ ಮಾಹಿತಿ",

        "nitrogen":
            "ನೈಟ್ರಜನ (N)",

        "phosphorus":
            "ಫಾಸ್ಫರಸ್ (P)",

        "potassium":
            "ಪೊಟ್ಯಾಸಿಯಮ್ (K)",

        "ph":
            "pH",

        "p2o5":
            "ಫಾಸ್ಫರಸ್ (P₂O₅)",

        "k2o":
            "ಪೊಟ್ಯಾಸಿಯಮ್ (K₂O)",

        "temperature":
            "ತಾಪಮಾನ (°C)",

        "humidity":
            "ಆರ್ದ್ರತೆ (%)",

        "rainfall":
            "ಮಳೆ (mm)",

        "manual":
            "🌱 ಇತರ ಮಾಹಿತಿಯನ್ನು ನಮೂದಿಸಿ",

        "soil_information":
            "🌱 ಮಣ್ಣಿನ ಮಾಹಿತಿ",

        "button":
            "ಬೆಳೆಗಳನ್ನು ಸೂಚಿಸಿ",

        "crop_title":
            "🌾 ಬೆಳೆಯಲು ಸೂಕ್ತವಾದ ಪ್ರಮುಖ ಬೆಳೆಗಳು",

        "not_detected":
            "ಪತ್ತೆಯಾಗಿಲ್ಲ",

        "uploaded":
            "ಅಪ್ಲೋಡ್ ಮಾಡಿದ ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್",

        "probability":
            "ಮಾದರಿ ಸಂಭವನೀಯತೆ",

        "no_crop":
            "ಈ ಪರಿಸ್ಥಿತಿಗಳಿಗೆ ಕನಿಷ್ಠ 10% ಸಂಭವನೀಯತೆಯ ಬೆಳೆ ಇಲ್ಲ.",

        "dataset_error":
            "Crop_recommendation.csv ಕಂಡುಬಂದಿಲ್ಲ. ದಯವಿಟ್ಟು ಅದನ್ನು crop_chatbot.py ಇರುವ ಅದೇ ಫೋಲ್ಡರ್‌ನಲ್ಲಿ ಇರಿಸಿ.",

        "missing_columns":
            "ಕ್ರಾಪ್ ಡೇಟಾಸೆಟ್‌ನಲ್ಲಿ ಈ ಕಾಲಮ್‌ಗಳು ಕಾಣೆಯಾಗಿವೆ:",

        "nitrogen_extracted":
            "ನೈಟ್ರಜನ (N)",

        "phosphorus_extracted":
            "ಫಾಸ್ಫರಸ್ (P₂O₅)",

        "potassium_extracted":
            "ಪೊಟ್ಯಾಸಿಯಮ್ (K₂O)"
    },


    # =====================================================
    # PUNJABI
    # =====================================================

    "Punjabi": {

        "title":
            "ਕ੍ਰਿਸ਼ਿਵੇਦ",

        "choose_language":
            "🌐 ਭਾਸ਼ਾ ਚੁਣੋ",

        "upload":
            "📎 KVK (ਕ੍ਰਿਸ਼ੀ ਵਿਗਿਆਨ ਕੇਂਦਰ) / ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਅਪਲੋਡ ਕਰੋ",

        "upload_help":
            "ਆਪਣੇ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਦੀ ਸਾਫ਼ ਤਸਵੀਰ ਅਪਲੋਡ ਕਰੋ",

        "reading":
            "ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ ਪੜ੍ਹਿਆ ਜਾ ਰਿਹਾ ਹੈ...",

        "extracted":
            "📄 ਤੁਹਾਡੀ ਰਿਪੋਰਟ ਤੋਂ ਪ੍ਰਾਪਤ ਜਾਣਕਾਰੀ",

        "nitrogen":
            "ਨਾਈਟ੍ਰੋਜਨ (N)",

        "phosphorus":
            "ਫਾਸਫੋਰਸ (P)",

        "potassium":
            "ਪੋਟਾਸ਼ੀਅਮ (K)",

        "ph":
            "pH",

        "p2o5":
            "ਫਾਸਫੋਰਸ (P₂O₅)",

        "k2o":
            "ਪੋਟਾਸ਼ੀਅਮ (K₂O)",

        "temperature":
            "ਤਾਪਮਾਨ (°C)",

        "humidity":
            "ਨਮੀ (%)",

        "rainfall":
            "ਵਰਖਾ (mm)",

        "manual":
            "🌱 ਹੋਰ ਜਾਣਕਾਰੀ ਦਰਜ ਕਰੋ",

        "soil_information":
            "🌱 ਮਿੱਟੀ ਦੀ ਜਾਣਕਾਰੀ",

        "button":
            "ਫਸਲਾਂ ਦੀ ਸਿਫਾਰਸ਼ ਕਰੋ",

        "crop_title":
            "🌾 ਉਗਾਉਣ ਲਈ ਮੁੱਖ ਫਸਲਾਂ",

        "not_detected":
            "ਪਤਾ ਨਹੀਂ ਲੱਗਿਆ",

        "uploaded":
            "ਅਪਲੋਡ ਕੀਤਾ ਮਿੱਟੀ ਸਿਹਤ ਕਾਰਡ",

        "probability":
            "ਮਾਡਲ ਸੰਭਾਵਨਾ",

        "no_crop":
            "ਇਨ੍ਹਾਂ ਹਾਲਾਤਾਂ ਲਈ ਕਿਸੇ ਵੀ ਫਸਲ ਦੀ ਸੰਭਾਵਨਾ ਘੱਟੋ-ਘੱਟ 10% ਨਹੀਂ ਹੈ।",

        "dataset_error":
            "Crop_recommendation.csv ਨਹੀਂ ਮਿਲੀ। ਕਿਰਪਾ ਕਰਕੇ ਇਸਨੂੰ crop_chatbot.py ਵਾਲੇ ਫੋਲਡਰ ਵਿੱਚ ਰੱਖੋ।",

        "missing_columns":
            "ਕ੍ਰਾਪ ਡੇਟਾਸੈੱਟ ਵਿੱਚ ਇਹ ਕਾਲਮ ਮੌਜੂਦ ਨਹੀਂ ਹਨ:",

        "nitrogen_extracted":
            "ਨਾਈਟ੍ਰੋਜਨ (N)",

        "phosphorus_extracted":
            "ਫਾਸਫੋਰਸ (P₂O₅)",

        "potassium_extracted":
            "ਪੋਟਾਸ਼ੀਅਮ (K₂O)"
    },


    # =====================================================
    # BENGALI
    # =====================================================

    "Bengali": {

        "title":
            "কৃষিবেদ",

        "choose_language":
            "🌐 ভাষা নির্বাচন করুন",

        "upload":
            "📎 KVK (কৃষি বিজ্ঞান কেন্দ্র) / মাটি স্বাস্থ্য কার্ড আপলোড করুন",

        "upload_help":
            "আপনার মাটি স্বাস্থ্য কার্ডের একটি পরিষ্কার ছবি আপলোড করুন",

        "reading":
            "মাটি স্বাস্থ্য কার্ড পড়া হচ্ছে...",

        "extracted":
            "📄 আপনার রিপোর্ট থেকে প্রাপ্ত তথ্য",

        "nitrogen":
            "নাইট্রোজেন (N)",

        "phosphorus":
            "ফসফরাস (P)",

        "potassium":
            "পটাশিয়াম (K)",

        "ph":
            "pH",

        "p2o5":
            "ফসফরাস (P₂O₅)",

        "k2o":
            "পটাশিয়াম (K₂O)",

        "temperature":
            "তাপমাত্রা (°C)",

        "humidity":
            "আর্দ্রতা (%)",

        "rainfall":
            "বৃষ্টি (mm)",

        "manual":
            "🌱 অন্যান্য তথ্য দিন",

        "soil_information":
            "🌱 মাটির তথ্য",

        "button":
            "ফসলের সুপারিশ করুন",

        "crop_title":
            "🌾 চাষের জন্য প্রধান ফসল",

        "not_detected":
            "পাওয়া যায়নি",

        "uploaded":
            "আপলোড করা মাটি স্বাস্থ্য কার্ড",

        "probability":
            "মডেল সম্ভাবনা",

        "no_crop":
            "এই পরিস্থিতিতে কোনো ফসলের সম্ভাবনা কমপক্ষে 10% নয়।",

        "dataset_error":
            "Crop_recommendation.csv পাওয়া যায়নি। অনুগ্রহ করে এটি crop_chatbot.py-এর একই ফোল্ডারে রাখুন।",

        "missing_columns":
            "ক্রপ ডেটাসেটে নিম্নলিখিত কলামগুলি নেই:",

        "nitrogen_extracted":
            "নাইট্রোজেন (N)",

        "phosphorus_extracted":
            "ফসফরাস (P₂O₅)",

        "potassium_extracted":
            "পটাশিয়াম (K₂O)"
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
# CROP TRANSLATION FUNCTION
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
    "🌐 Choose Language",
    list(lang_codes.keys())
)


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
        labels[language]["title"]
    )


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

            except Exception:

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

            except Exception:

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

            except Exception:

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

            except Exception:

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
        labels[language]["reading"]
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
                f"**{labels[language]['nitrogen_extracted']}:** "
                f"{extracted_N}"
            )

        else:

            st.write(
                f"**{labels[language]['nitrogen_extracted']}:** "
                f"{labels[language]['not_detected']}"
            )


        if extracted_P2O5 is not None:

            st.write(
                f"**{labels[language]['phosphorus_extracted']}:** "
                f"{extracted_P2O5}"
            )

        else:

            st.write(
                f"**{labels[language]['phosphorus_extracted']}:** "
                f"{labels[language]['not_detected']}"
            )


    with col2:

        if extracted_K2O is not None:

            st.write(
                f"**{labels[language]['potassium_extracted']}:** "
                f"{extracted_K2O}"
            )

        else:

            st.write(
                f"**{labels[language]['potassium_extracted']}:** "
                f"{labels[language]['not_detected']}"
            )


        if extracted_pH is not None:

            st.write(
                f"**{labels[language]['ph']}:** "
                f"{extracted_pH}"
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
        labels[language]["dataset_error"]
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
        f"{labels[language]['missing_columns']} "
        f"{missing_crop_columns}"
    )

    st.stop()


# =========================================================
# TRAINING DATA
# =========================================================

X = data[
    required_crop_columns
]

y = data[
    "label"
]


# =========================================================
# RANDOM FOREST CLASSIFIER
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X,
    y
)


# =========================================================
# CROP PREDICTION
# =========================================================

if st.button(
    labels[language]["button"],
    type="primary"
):

    # =====================================================
    # IMPORTANT:
    # FEATURE ORDER MUST MATCH TRAINING ORDER
    # =====================================================

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


    # =====================================================
    # PREDICT PROBABILITIES
    # =====================================================

    probabilities = model.predict_proba(
        crop_input
    )[0]


    crop_names = model.classes_


    # =====================================================
    # SORT HIGHEST → LOWEST
    # =====================================================

    sorted_indices = np.argsort(
        probabilities
    )[::-1]


    # =====================================================
    # FILTER ONLY ≥10%
    # =====================================================

    valid_indices = []


    for index in sorted_indices:

        probability_percent = (
            probabilities[index] * 100
        )

        if probability_percent >= 10.0:

            valid_indices.append(
                index
            )


    # =====================================================
    # MAXIMUM 3 CROPS
    # =====================================================

    top_indices = valid_indices[:3]


    # =====================================================
    # HEADING
    # =====================================================

    st.subheader(
        labels[language]["crop_title"]
    )


    # =====================================================
    # NO CROPS ≥10%
    # =====================================================

    if len(top_indices) == 0:

        st.warning(
            labels[language]["no_crop"]
        )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        crop_name = crop_names[index]


        probability = (
            probabilities[index] * 100
        )


        # =================================================
        # TRANSLATE CROP NAME
        # =================================================

        translated_crop = translate_crop_name(
            crop_name,
            lang_codes[language]
        )


        # =================================================
        # RANK 1
        # =================================================

        if rank == 1:

            st.success(
                f"🥇 {translated_crop}"
            )


        # =================================================
        # RANK 2
        # =================================================

        elif rank == 2:

            st.info(
                f"🥈 {translated_crop}"
            )


        # =================================================
        # RANK 3
        # =================================================

        elif rank == 3:

            st.warning(
                f"🥉 {translated_crop}"
            )


        # =================================================
        # PROBABILITY
        # =================================================

        st.write(
            f"{labels[language]['probability']}: "
            f"{probability:.2f}%"
        )