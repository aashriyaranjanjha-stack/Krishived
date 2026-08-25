import pandas as pd
import streamlit as st
import pytesseract
import cv2
import re
import numpy as np

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

        "nitrogen":
            "Nitrogen (N)",

        "phosphorus":
            "Phosphorus (P)",

        "potassium":
            "Potassium (K)",

        "ph":
            "pH",

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
            "Phosphorus (P)",

        "potassium_extracted":
            "Potassium (K)"
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

        "nitrogen":
            "नाइट्रोजन (N)",

        "phosphorus":
            "फॉस्फोरस (P)",

        "potassium":
            "पोटैशियम (K)",

        "ph":
            "pH",

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
            "फॉस्फोरस (P)",

        "potassium_extracted":
            "पोटैशियम (K)"
    },


    "Kannada": {

        "title": "ಕೃಷಿವೇದ",

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
            "ಫಾಸ್ಫರಸ್ (P)",

        "potassium_extracted":
            "ಪೊಟ್ಯಾಸಿಯಮ್ (K)"
    },


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
            "ਫਾਸਫੋਰਸ (P)",

        "potassium_extracted":
            "ਪੋਟਾਸ਼ੀਅਮ (K)"
    },


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
            "ফসফরাস (P)",

        "potassium_extracted":
            "পটাশিয়াম (K)"
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
    labels["English"]["choose_language"],
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
# SOIL HEALTH CARD OCR
# =========================================================

def detect_green_status_centers(img):

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    lower_green = np.array(
        [35, 45, 70]
    )

    upper_green = np.array(
        [95, 255, 255]
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    num_labels, label_image, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            8
        )
    )

    centers = []

    for i in range(
        1,
        num_labels
    ):

        x, y, w, h, area = stats[i]

        if (
            25 <= w <= 65
            and
            25 <= h <= 65
            and
            500 <= area <= 4000
            and
            0.75 <= (w / h) <= 1.25
        ):

            cx, cy = centroids[i]

            centers.append(
                (
                    float(cx),
                    float(cy)
                )
            )

    return centers


# =========================================================
# CLUSTER POSITIONS
# =========================================================

def cluster_positions(
    values,
    tolerance=60
):

    if not values:

        return []

    values = sorted(
        values
    )

    clusters = []

    for value in values:

        if not clusters:

            clusters.append(
                [value]
            )

            continue

        current_average = np.mean(
            clusters[-1]
        )

        if abs(
            value - current_average
        ) <= tolerance:

            clusters[-1].append(
                value
            )

        else:

            clusters.append(
                [value]
            )

    return [
        float(
            np.mean(cluster)
        )
        for cluster in clusters
    ]


# =========================================================
# DETECT CARD GRID
# =========================================================

def detect_card_grid(img):

    centers = detect_green_status_centers(
        img
    )

    if len(centers) < 4:

        return None, None

    x_positions = [
        point[0]
        for point in centers
    ]

    y_positions = [
        point[1]
        for point in centers
    ]

    columns = cluster_positions(
        x_positions,
        tolerance=60
    )

    rows = cluster_positions(
        y_positions,
        tolerance=60
    )

    if len(columns) < 3:

        return None, None

    if len(rows) < 2:

        return None, None

    columns = sorted(
        columns
    )[:3]

    rows = sorted(
        rows
    )[:4]

    return columns, rows


# =========================================================
# GET NUTRIENT BOX
# =========================================================

def get_nutrient_box(
    img,
    columns,
    rows,
    column_index,
    row_index
):

    height, width = img.shape[:2]

    dx = np.median(
        np.diff(columns)
    )

    if row_index == 0:

        dy = (
            rows[1]
            -
            rows[0]
        )

    elif row_index == len(rows) - 1:

        dy = (
            rows[-1]
            -
            rows[-2]
        )

    else:

        dy = (
            rows[row_index + 1]
            -
            rows[row_index - 1]
        ) / 2.0

    cx = columns[
        column_index
    ]

    cy = rows[
        row_index
    ]

    x1 = int(
        cx - 0.825 * dx
    )

    x2 = int(
        cx + 0.17 * dx
    )

    y1 = int(
        cy - 0.48 * dy
    )

    y2 = int(
        cy + 0.47 * dy
    )

    x1 = max(
        0,
        min(
            width - 1,
            x1
        )
    )

    x2 = max(
        x1 + 1,
        min(
            width,
            x2
        )
    )

    y1 = max(
        0,
        min(
            height - 1,
            y1
        )
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            y2
        )
    )

    return img[
        y1:y2,
        x1:x2
    ]


# =========================================================
# NORMALIZE OCR NUMBER
# =========================================================

def normalize_ocr_number(
    raw_number,
    nutrient
):

    try:

        value = float(
            raw_number
        )

    except:

        return None

    if nutrient == "pH":

        if 3.5 <= value <= 9.9:

            return value

        if (
            value >= 100
            and
            len(
                re.sub(
                    r"\D",
                    "",
                    raw_number
                )
            ) == 3
        ):

            corrected = value / 100.0

            if 3.5 <= corrected <= 9.9:

                return corrected

        return None

    if nutrient == "N":

        if 50 <= value <= 1000:

            return value

        if (
            value.is_integer()
            and
            4 <= len(
                re.sub(
                    r"\D",
                    "",
                    raw_number
                )
            ) <= 5
        ):

            corrected = value / 100.0

            if 50 <= corrected <= 1000:

                return corrected

        return None

    if nutrient == "P":

        if 0.1 <= value <= 500:

            return value

        if (
            value.is_integer()
            and
            4 <= len(
                re.sub(
                    r"\D",
                    "",
                    raw_number
                )
            ) <= 5
        ):

            corrected = value / 100.0

            if 0.1 <= corrected <= 500:

                return corrected

        return None

    if nutrient == "K":

        if 0.1 <= value <= 1000:

            return value

        if (
            value.is_integer()
            and
            4 <= len(
                re.sub(
                    r"\D",
                    "",
                    raw_number
                )
            ) <= 5
        ):

            corrected = value / 100.0

            if 0.1 <= corrected <= 1000:

                return corrected

        return None

    return None


# =========================================================
# OCR A NUTRIENT BOX
# =========================================================

def ocr_nutrient_box(
    crop,
    nutrient
):

    if crop is None:

        return None

    if crop.size == 0:

        return None

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.normalize(
        gray,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    processed_images = []

    processed_images.append(
        gray
    )

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        +
        cv2.THRESH_OTSU
    )

    processed_images.append(
        otsu
    )

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    processed_images.append(
        adaptive
    )

    candidates = []

    for processed in processed_images:

        text = pytesseract.image_to_string(
            processed,
            config="--psm 6"
        )

        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            text
        )

        for number in numbers:

            corrected = normalize_ocr_number(
                number,
                nutrient
            )

            if corrected is not None:

                candidates.append(
                    corrected
                )

        numeric_text = pytesseract.image_to_string(
            processed,
            config=(
                "--psm 6 "
                "-c tessedit_char_whitelist="
                "0123456789."
            )
        )

        numeric_numbers = re.findall(
            r"\d+(?:\.\d+)?",
            numeric_text
        )

        for number in numeric_numbers:

            corrected = normalize_ocr_number(
                number,
                nutrient
            )

            if corrected is not None:

                candidates.append(
                    corrected
                )

    if not candidates:

        return None

    rounded_candidates = [
        round(
            value,
            2
        )
        for value in candidates
    ]

    counts = {}

    for value in rounded_candidates:

        counts[value] = (
            counts.get(
                value,
                0
            )
            +
            1
        )

    best_value = max(
        counts,
        key=counts.get
    )

    return float(
        best_value
    )


# =========================================================
# EXTRACT SOIL VALUES
# =========================================================

def extract_soil_values(image):

    img = np.array(
        image
    )

    values = {
        "N": None,
        "P": None,
        "K": None,
        "pH": None
    }

    columns, rows = detect_card_grid(
        img
    )

    if (
        columns is not None
        and
        rows is not None
        and
        len(columns) >= 3
        and
        len(rows) >= 2
    ):

        nitrogen_crop = get_nutrient_box(
            img,
            columns,
            rows,
            0,
            0
        )

        values["N"] = ocr_nutrient_box(
            nitrogen_crop,
            "N"
        )

        phosphorus_crop = get_nutrient_box(
            img,
            columns,
            rows,
            1,
            0
        )

        values["P"] = ocr_nutrient_box(
            phosphorus_crop,
            "P"
        )

        potassium_crop = get_nutrient_box(
            img,
            columns,
            rows,
            2,
            0
        )

        values["K"] = ocr_nutrient_box(
            potassium_crop,
            "K"
        )

        ph_crop = get_nutrient_box(
            img,
            columns,
            rows,
            0,
            1
        )

        values["pH"] = ocr_nutrient_box(
            ph_crop,
            "pH"
        )

    else:

        height, width = img.shape[:2]

        nitrogen_crop = img[
            int(height * 0.07):
            int(height * 0.25),
            int(width * 0.02):
            int(width * 0.38)
        ]

        values["N"] = ocr_nutrient_box(
            nitrogen_crop,
            "N"
        )

        phosphorus_crop = img[
            int(height * 0.07):
            int(height * 0.25),
            int(width * 0.37):
            int(width * 0.66)
        ]

        values["P"] = ocr_nutrient_box(
            phosphorus_crop,
            "P"
        )

        potassium_crop = img[
            int(height * 0.07):
            int(height * 0.25),
            int(width * 0.64):
            int(width * 0.96)
        ]

        values["K"] = ocr_nutrient_box(
            potassium_crop,
            "K"
        )

        ph_crop = img[
            int(height * 0.22):
            int(height * 0.42),
            int(width * 0.02):
            int(width * 0.38)
        ]

        values["pH"] = ocr_nutrient_box(
            ph_crop,
            "pH"
        )

    return values


# =========================================================
# UPLOAD SOIL HEALTH CARD
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
# PROCESS UPLOADED REPORT
# =========================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption=labels[language]["uploaded"],
        use_container_width=True
    )

    with st.spinner(
        labels[language]["reading"]
    ):

        soil_values = extract_soil_values(
            image
        )

    extracted_N = soil_values["N"]
    extracted_P = soil_values["P"]
    extracted_K = soil_values["K"]
    extracted_pH = soil_values["pH"]

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
# NITROGEN
# =========================================================

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


# =========================================================
# PHOSPHORUS
# =========================================================

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


# =========================================================
# POTASSIUM
# =========================================================

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
# CHECK LABEL COLUMN
# =========================================================

if "label" not in data.columns:

    st.error(
        "Crop dataset must contain a 'label' column."
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
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
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


# =========================================================
# TRAIN MODEL ONLY ON TRAINING DATA
# =========================================================

model.fit(
    X_train,
    y_train
)


# =========================================================
# TEST MODEL
# =========================================================

y_test_pred = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)


# =========================================================
# CROP PREDICTION
# =========================================================

if st.button(
    labels[language]["button"],
    type="primary"
):

    # =====================================================
    # NITROGEN CONVERSION
    #
    # Displayed:
    # N = 528
    #
    # Model:
    # N = 528 × 0.30
    #
    # Only model input is changed.
    # =====================================================

    N_for_model = N * 0.30


    # =====================================================
    # CREATE MODEL INPUT
    # =====================================================

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