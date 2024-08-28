import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from transformers import BertTokenizer, BertModel
import re
import speech_recognition as sr

# Yüklenecek model ve yardımcı araçların yolları
SENTIMENT_MODEL_PATH = 'mlp_model.pkl'
AUDIO_MODEL_PATH = 'model10.h5'
ENCODER_PATH = 'encoder.pkl'
SCALER_PATH = 'scaler.pkl'
STOP_WORDS_PATH = 'stop.txt'
TEMP_AUDIO_PATH = 'temp.wav'

# Mikrofon özellikleri
SAMPLE_RATE = 44100  # Örnekleme oranı (Hz)

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

def predict_emotion_from_audio(file_path):
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    features = extract_features_from_audio(audio, SAMPLE_RATE)
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
    metin = ' '.join([kelime için kelime in metin.split() if len(kelime) >= 3])
    temiz_kelimeler = [kelime for kelime in metin.split() if kelime not in stop_words]
    return ' '.join(temiz_kelimeler)

# Ses kaydını hem ses modelinden hem metin modelinden geçir
def process_audio():
    # Ses modelinden tahmin yap
    audio_predictions = predict_emotion_from_audio(TEMP_AUDIO_PATH)
    
    # Metne dönüştürme
    recognizer = sr.Recognizer()
    with sr.AudioFile(TEMP_AUDIO_PATH) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language='tr-TR')
    except sr.UnknownValueError:
        text = None
    except sr.RequestError:
        text = None

    if text:
        # Metin modelinden tahmin yap
        cleaned_text = temizle_metin(text, stop_words)
        text_vector = metni_vektore_cevir(cleaned_text, tokenizer, bert_model)
        text_vector = text_vector.reshape(1, -1)
        text_prediction = mlp_model.predict(text_vector)
        text_prediction = text_prediction[0]
    else:
        text_prediction = None
    
    return audio_predictions, text_prediction

# Nihai tahmin hesaplama
def get_final_prediction(audio_predictions, text_prediction):
    if text_prediction is not None:
        final_prediction = text_prediction
    else:
        final_prediction = max(audio_predictions, key=audio_predictions.get)
    
    # Tahmini "angry" ve "normal" yerine "negatif" ve "pozitif" olarak döndürme
    if final_prediction == "angry":
        final_prediction = "negatif"
    elif final_prediction == "normal":
        final_prediction = "pozitif"
    
    return final_prediction

# Streamlit arayüzü
st.title("Ses Tabanlı Duygu Tanıma")
st.write("Bir ses dosyası yükleyin ya da mikrofon ile ses kaydedin ve duygu tahminini görün.")

uploaded_file = st.file_uploader("Bir ses dosyası yükleyin", type=["wav"])
if uploaded_file is not None:
    with open(TEMP_AUDIO_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    audio_predictions, text_prediction = process_audio()
    final_prediction = get_final_prediction(audio_predictions, text_prediction)
    st.write(f"Tahmin: {final_prediction}")
    st.write(f"Ses Modeli Tahminleri: {audio_predictions}")
    if text_prediction is not None:
        st.write(f"Metin Modeli Tahmini: {text_prediction}")

# Ses kaydını başlatmak için bir buton
if st.button("Mikrofonla Ses Kaydı Yap"):
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
        st.write("Ses kaydediliyor...")
        audio = recognizer.listen(source, phrase_time_limit=5)
        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio.get_wav_data())
    
    audio_predictions, text_prediction = process_audio()
    final_prediction = get_final_prediction(audio_predictions, text_prediction)
    st.write(f"Tahmin: {final_prediction}")
    st.write(f"Ses Modeli Tahminleri: {audio_predictions}")
    if text_prediction is not None:
        st.write(f"Metin Modeli Tahmini: {text_prediction}")
