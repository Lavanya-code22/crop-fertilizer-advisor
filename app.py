import streamlit as st
import pandas as pd
import joblib

model = joblib.load("crop_model.pkl")
ideal = pd.read_csv("ideal_npk.csv", index_col=0)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

TEXT = {
    "English": {
        "title": "Crop and Fertilizer Advisor",
        "intro": "Enter your soil and weather values to get a crop recommendation.",
        "city": "City (optional, to auto-fill weather)",
        "fetch": "Fetch weather",
        "N": "Nitrogen (N)",
        "P": "Phosphorus (P)",
        "K": "Potassium (K)",
        "temp": "Temperature (°C)",
        "hum": "Humidity (%)",
        "ph": "Soil pH",
        "rain": "Rainfall (mm)",
        "button": "Get Recommendation",
        "top3": "Top 3 crops",
        "fert": "Fertilizer advice for",
        "why": "Why this recommendation?",
        "warn": "The model is not very sure about this result. Check your soil values or ask a local agriculture officer.",
        "download": "Download PDF report",
        "relies": "The model relies most on:",
    },
    "తెలుగు": {
        "title": "పంట మరియు ఎరువుల సలహాదారు",
        "intro": "పంట సిఫార్సు పొందడానికి మీ నేల మరియు వాతావరణ విలువలను నమోదు చేయండి.",
        "city": "నగరం (ఐచ్ఛికం, వాతావరణం నింపడానికి)",
        "fetch": "వాతావరణం తీసుకురండి",
        "N": "నత్రజని (N)",
        "P": "భాస్వరం (P)",
        "K": "పొటాషియం (K)",
        "temp": "ఉష్ణోగ్రత (°C)",
        "hum": "తేమ (%)",
        "ph": "నేల pH",
        "rain": "వర్షపాతం (మి.మీ)",
        "button": "సిఫార్సు పొందండి",
        "top3": "టాప్ 3 పంటలు",
        "fert": "ఎరువుల సలహా:",
        "why": "ఈ సిఫార్సు ఎందుకు?",
        "warn": "ఈ ఫలితంపై మోడల్‌కు పూర్తి నమ్మకం లేదు. మీ నేల విలువలను తనిఖీ చేయండి లేదా స్థానిక వ్యవసాయ అధికారిని అడగండి.",
                "download": "PDF నివేదికను డౌన్‌లోడ్ చేయండి",
                "relies": "మోడల్ ఎక్కువగా ఆధారపడేవి:",
    },
    "हिन्दी": {
        "title": "फसल और उर्वरक सलाहकार",
        "intro": "फसल की सिफारिश पाने के लिए अपनी मिट्टी और मौसम के मान दर्ज करें।",
        "city": "शहर (वैकल्पिक, मौसम अपने आप भरने के लिए)",
        "fetch": "मौसम लाएं",
        "N": "नाइट्रोजन (N)",
        "P": "फॉस्फोरस (P)",
        "K": "पोटैशियम (K)",
        "temp": "तापमान (°C)",
        "hum": "नमी (%)",
        "ph": "मिट्टी का pH",
        "rain": "वर्षा (मिमी)",
        "button": "सिफारिश पाएं",
        "top3": "शीर्ष 3 फसलें",
        "fert": "उर्वरक सलाह:",
        "why": "यह सिफारिश क्यों?",
        "warn": "मॉडल को इस नतीजे पर पूरा भरोसा नहीं है। अपनी मिट्टी के मान जांचें या स्थानीय कृषि अधिकारी से पूछें।",
        "download": "PDF रिपोर्ट डाउनलोड करें",
        "relies": "मॉडल सबसे ज़्यादा इन पर निर्भर करता है:",
    },
    "ಕನ್ನಡ": {
        "title": "ಬೆಳೆ ಮತ್ತು ಗೊಬ್ಬರ ಸಲಹೆಗಾರ",
        "intro": "ಬೆಳೆ ಶಿಫಾರಸು ಪಡೆಯಲು ನಿಮ್ಮ ಮಣ್ಣು ಮತ್ತು ಹವಾಮಾನ ಮೌಲ್ಯಗಳನ್ನು ನಮೂದಿಸಿ.",
        "city": "ನಗರ (ಐಚ್ಛಿಕ, ಹವಾಮಾನ ತುಂಬಲು)",
        "fetch": "ಹವಾಮಾನ ತರಿ",
        "N": "ಸಾರಜನಕ (N)",
        "P": "ರಂಜಕ (P)",
        "K": "ಪೊಟ್ಯಾಸಿಯಮ್ (K)",
        "temp": "ತಾಪಮಾನ (°C)",
        "hum": "ತೇವಾಂಶ (%)",
        "ph": "ಮಣ್ಣಿನ pH",
        "rain": "ಮಳೆ (ಮಿ.ಮೀ)",
        "button": "ಶಿಫಾರಸು ಪಡೆಯಿರಿ",
        "top3": "ಅಗ್ರ 3 ಬೆಳೆಗಳು",
        "fert": "ಗೊಬ್ಬರ ಸಲಹೆ:",
        "why": "ಈ ಶಿಫಾರಸು ಏಕೆ?",
        "warn": "ಈ ಫಲಿತಾಂಶದ ಬಗ್ಗೆ ಮಾದರಿಗೆ ಪೂರ್ಣ ವಿಶ್ವಾಸವಿಲ್ಲ. ನಿಮ್ಮ ಮಣ್ಣಿನ ಮೌಲ್ಯಗಳನ್ನು ಪರಿಶೀಲಿಸಿ ಅಥವಾ ಸ್ಥಳೀಯ ಕೃಷಿ ಅಧಿಕಾರಿಯನ್ನು ಕೇಳಿ.",
        "download": "PDF ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",
        "relies": "ಮಾದರಿ ಹೆಚ್ಚಾಗಿ ಇವುಗಳ ಮೇಲೆ ಅವಲಂಬಿತವಾಗಿದೆ:",
    },
}

TIPS = {
    "తెలుగు": {
        "Nitrogen is low: add urea.": "నత్రజని తక్కువగా ఉంది: యూరియా వేయండి.",
        "Nitrogen is high: avoid adding nitrogen fertilizer.": "నత్రజని ఎక్కువగా ఉంది: నత్రజని ఎరువు వేయకండి.",
        "Phosphorus is low: add DAP or single super phosphate.": "భాస్వరం తక్కువగా ఉంది: DAP లేదా సింగిల్ సూపర్ ఫాస్ఫేట్ వేయండి.",
        "Phosphorus is high: avoid adding phosphorus fertilizer.": "భాస్వరం ఎక్కువగా ఉంది: భాస్వరం ఎరువు వేయకండి.",
        "Potassium is low: add muriate of potash (MOP).": "పొటాషియం తక్కువగా ఉంది: మ్యూరియేట్ ఆఫ్ పొటాష్ (MOP) వేయండి.",
        "Potassium is high: avoid adding potassium fertilizer.": "పొటాషియం ఎక్కువగా ఉంది: పొటాషియం ఎరువు వేయకండి.",
        "Soil nutrients are well balanced for this crop.": "ఈ పంటకు నేల పోషకాలు సమతుల్యంగా ఉన్నాయి.",
    },
    "हिन्दी": {
        "Nitrogen is low: add urea.": "नाइट्रोजन कम है: यूरिया डालें।",
        "Nitrogen is high: avoid adding nitrogen fertilizer.": "नाइट्रोजन ज़्यादा है: नाइट्रोजन खाद न डालें।",
        "Phosphorus is low: add DAP or single super phosphate.": "फॉस्फोरस कम है: DAP या सिंगल सुपर फॉस्फेट डालें।",
        "Phosphorus is high: avoid adding phosphorus fertilizer.": "फॉस्फोरस ज़्यादा है: फॉस्फोरस खाद न डालें।",
        "Potassium is low: add muriate of potash (MOP).": "पोटैशियम कम है: म्यूरेट ऑफ पोटाश (MOP) डालें।",
        "Potassium is high: avoid adding potassium fertilizer.": "पोटैशियम ज़्यादा है: पोटैशियम खाद न डालें।",
        "Soil nutrients are well balanced for this crop.": "इस फसल के लिए मिट्टी के पोषक तत्व संतुलित हैं।",
    },
    "ಕನ್ನಡ": {
        "Nitrogen is low: add urea.": "ಸಾರಜನಕ ಕಡಿಮೆ ಇದೆ: ಯೂರಿಯಾ ಹಾಕಿ.",
        "Nitrogen is high: avoid adding nitrogen fertilizer.": "ಸಾರಜನಕ ಹೆಚ್ಚು ಇದೆ: ಸಾರಜನಕ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
        "Phosphorus is low: add DAP or single super phosphate.": "ರಂಜಕ ಕಡಿಮೆ ಇದೆ: DAP ಅಥವಾ ಸಿಂಗಲ್ ಸೂಪರ್ ಫಾಸ್ಫೇಟ್ ಹಾಕಿ.",
        "Phosphorus is high: avoid adding phosphorus fertilizer.": "ರಂಜಕ ಹೆಚ್ಚು ಇದೆ: ರಂಜಕ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
        "Potassium is low: add muriate of potash (MOP).": "ಪೊಟ್ಯಾಸಿಯಮ್ ಕಡಿಮೆ ಇದೆ: ಮ್ಯೂರಿಯೇಟ್ ಆಫ್ ಪೊಟ್ಯಾಶ್ (MOP) ಹಾಕಿ.",
        "Potassium is high: avoid adding potassium fertilizer.": "ಪೊಟ್ಯಾಸಿಯಮ್ ಹೆಚ್ಚು ಇದೆ: ಪೊಟ್ಯಾಸಿಯಮ್ ಗೊಬ್ಬರ ಹಾಕಬೇಡಿ.",
        "Soil nutrients are well balanced for this crop.": "ಈ ಬೆಳೆಗೆ ಮಣ್ಣಿನ ಪೋಷಕಾಂಶಗಳು ಸಮತೋಲನದಲ್ಲಿವೆ.",
    },
}

import requests


def get_weather(city):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
        timeout=10,
    ).json()
    if "results" not in geo:
        return None
    place = geo["results"][0]

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m",
        },
        timeout=10,
    ).json()
    cur = weather["current"]
    return place["name"], cur["temperature_2m"], cur["relative_humidity_2m"]
from fpdf import FPDF


def make_pdf(inputs, top3_list, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Crop and Fertilizer Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Your inputs", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for name, value in inputs.items():
        pdf.cell(0, 7, f"{name}: {value}", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Top 3 crops", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for crop, pct in top3_list:
        pdf.cell(0, 7, f"{crop}: {pct:.1f}%", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Fertilizer advice", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    for tip in tips:
        pdf.multi_cell(0, 7, "- " + tip, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, "This is a decision-support tool, not a replacement for a soil test or a local agriculture officer.")
    return bytes(pdf.output())


def fertilizer_advice(crop, N, P, K):
    target = ideal.loc[crop]
    tips = []

    if N < target["N"] * 0.8:
        tips.append("Nitrogen is low: add urea.")
    elif N > target["N"] * 1.2:
        tips.append("Nitrogen is high: avoid adding nitrogen fertilizer.")

    if P < target["P"] * 0.8:
        tips.append("Phosphorus is low: add DAP or single super phosphate.")
    elif P > target["P"] * 1.2:
        tips.append("Phosphorus is high: avoid adding phosphorus fertilizer.")

    if K < target["K"] * 0.8:
        tips.append("Potassium is low: add muriate of potash (MOP).")
    elif K > target["K"] * 1.2:
        tips.append("Potassium is high: avoid adding potassium fertilizer.")

    if not tips:
        tips.append("Soil nutrients are well balanced for this crop.")
    return tips


lang = st.selectbox("Language / भाषा / భాష / ಭಾಷೆ", list(TEXT.keys()))
T = TEXT[lang]

st.title(T["title"])
st.write(T["intro"])
if "temp" not in st.session_state:
    st.session_state.temp = 25.0
    st.session_state.hum = 70.0

city = st.text_input(T["city"])
if st.button(T["fetch"]):
    result = get_weather(city)
    if result is None:
        st.error("City not found. Check the spelling.")
    else:
        name, t, h = result
        st.session_state.temp = float(t)
        st.session_state.hum = float(h)
        st.success(f"Weather for {name}: {t}°C, {h}% humidity")

N = st.number_input(T["N"], 0, 140, 50)
P = st.number_input(T["P"], 5, 145, 50)
K = st.number_input(T["K"], 5, 205, 50)
temperature = st.number_input(T["temp"], 8.0, 44.0, min(max(st.session_state.temp, 8.0), 44.0))
humidity = st.number_input(T["hum"], 14.0, 100.0, min(max(st.session_state.hum, 14.0), 100.0))
ph = st.number_input(T["ph"], 3.5, 9.9, 6.5)
rainfall = st.number_input(T["rain"], 20.0, 300.0, 100.0)

if st.button(T["button"]):
    row = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
    probs = model.predict_proba(row)[0]
    top3 = probs.argsort()[::-1][:3]

    st.subheader(T["top3"])
    for i in top3:
        st.write(f"**{model.classes_[i]}**: {float(probs[i]) * 100:.1f}%")

    best_crop = model.classes_[top3[0]]
    if probs[top3[0]] < 0.5:
       st.warning(T["warn"])
    st.subheader(f"{T['fert']} {best_crop}")
    for tip in fertilizer_advice(best_crop, N, P, K):
        shown = TIPS.get(lang, {}).get(tip, tip)
        st.write("- " + shown)
    st.subheader(T["why"])
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    top_names = ", ".join(importance.index[:3])
    st.write(T["relies"] + " " + top_names + ".")
    st.bar_chart(importance)
    pdf_bytes = make_pdf(
    {"N": N, "P": P, "K": K, "Temperature": temperature,
        "Humidity": humidity, "pH": ph, "Rainfall": rainfall},
    [(model.classes_[i], float(probs[i]) * 100) for i in top3],
    fertilizer_advice(best_crop, N, P, K),
    )
    st.download_button(T["download"], pdf_bytes, "crop_report.pdf", "application/pdf")