import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from transformers import BertTokenizer, BertModel
import re
import speech_recognition as sr
import streamlit as st

# Yüklenecek model ve yardımcı araçların yolları
SENTIMENT_MODEL_PATH = 'mlp_model.pkl'
AUDIO_MODEL_PATH = 'model10.h5'
ENCODER_PATH = 'encoder.pkl'
SCALER_PATH = 'scaler.pkl'
STOP_WORDS_PATH = 'stop.txt'

# BERT Tokenizer ve Modeli Yükle
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
bert_model = BertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

# Önceden eğitilmiş modeli ve yardımcı araçları yükleme
audio_model = load_model(AUDIO_MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
mlp_model = joblib.load(SENTIMENT_MODEL_PATH)

# Stop words yükleme
def load_stop_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        stop_words = set(word.strip().lower() for word in file.readlines())
    return stop_words

stop_words = load_stop_words(STOP_WORDS_PATH)

# Sesten özellik çıkarımı
def extract_features_from_audio(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result

def predict_emotion_from_audio(data, sample_rate):
    features = extract_features_from_audio(data, sample_rate)
    features = scaler.transform([features])
    features = np.expand_dims(features, axis=2)
    predictions = audio_model.predict(features)
    prediction_percentages = predictions[0] * 100
    emotion_labels = encoder.categories_[0]
    return dict(zip(emotion_labels, prediction_percentages))

# Metni temizleme ve vektöre çevirme
def metni_vektore_cevir(metin, tokenizer, model, max_length=128):
    inputs = tokenizer(metin, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

def temizle_metin(metin, stop_words):
    if not isinstance(metin, str):
        return ''
    metin = metin.lower().strip()
    metin = re.sub(r'\s+', ' ', metin)
    metin = metin.replace("''", '"').replace("'", '"')
    metin = re.sub(r'[0-9.,!?]', '', metin)
    metin = re.sub(r'[^a-zA-ZçğıöşüÇĞİÖŞÜ\s]', '', metin)
    metin = ' '.join([kelime for kelime in metin.split() if len(kelime) >= 3])
    temiz_kelimeler = [kelime for kelime in metin.split() if kelime not in stop_words]
    return ' '.join(temiz_kelimeler)

def anlamsiz_kelime_tespit_et(metin, min_kelime_uzunlugu=4):
    if not isinstance(metin, str):
        return ''
    kelimeler = metin.split()
    temiz_kelimeler = [kelime for kelime in kelimeler if len(kelime) >= min_kelime_uzunlugu]
    return ' '.join(temiz_kelimeler)

# Ses modelinin etiketlerini metin modelinin etiketlerine dönüştürme
def convert_labels(audio_labels):
    label_conversion = {
        'normal': 'normal',
        'fear': 'normal',
        'angry': 'angry'
    }
    
    converted_labels = {label_conversion.get(label, label): prob for label, prob in audio_labels.items()}
    return converted_labels

# Ses kaydını hem ses modelinden hem metin modelinden geçir
def process_audio(audio_file):
    # Streamlit'in yüklediği dosyayı geçici bir dosya olarak kaydetme
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Ses modelinden tahmin yap
    audio, sample_rate = librosa.load(temp_audio_path, sr=None)
    audio_predictions = predict_emotion_from_audio(audio, sample_rate)
    audio_predictions = convert_labels(audio_predictions)  # Etiketleri dönüştür

    # Metne dönüştürme
    recognizer = sr.Recognizer()
    with sr.AudioFile(temp_audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data, language='tr-TR')
    except sr.UnknownValueError:
        text = None
    except sr.RequestError:
        text = None

    if text:
        # Metin modelinden tahmin yap
        cleaned_text = temizle_metin(text, stop_words)
        cleaned_text = anlamsiz_kelime_tespit_et(cleaned_text)
        text_vector = metni_vektore_cevir(cleaned_text, tokenizer, bert_model)
        text_vector = text_vector.reshape(1, -1)
        text_prediction = mlp_model.predict_proba(text_vector)[0]
    else:
        text_prediction = None
    
    return audio_predictions, text_prediction

# Nihai tahmin hesaplama
def get_final_prediction(audio_predictions, text_prediction, text_weight=0.7, audio_weight=0.3):
    if text_prediction is not None:
        # Nihai tahmini hesaplama
        final_prediction = {}
        for label in audio_predictions.keys():
            if label in text_prediction:
                final_prediction[label] = (text_prediction[label] * text_weight) + (audio_predictions[label] * audio_weight)
            else:
                final_prediction[label] = audio_predictions[label] * audio_weight
        
        final_prediction = max(final_prediction, key=final_prediction.get)
    else:
        final_prediction = max(audio_predictions, key=audio_predictions.get)
    
    # Tahmini "angry" ve "normal" yerine "negatif" ve "pozitif" olarak döndürme
    if final_prediction == "angry":
        final_prediction = "negatif"
    elif final_prediction == "normal":
        final_prediction = "pozitif"
    
    return final_prediction

# Streamlit arayüzü
st.title("Duygu Analizi ve Ses Tanıma")
st.write("Bir ses dosyası yükleyin ve analiz edin.")

audio_file = st.file_uploader("Ses Dosyasını Yükleyin", type=["wav"])

if audio_file is not None:
    audio_predictions, text_prediction = process_audio(audio_file)
    
    final_prediction = get_final_prediction(audio_predictions, text_prediction)
    st.write(f"\nTahmin: {final_prediction}")
