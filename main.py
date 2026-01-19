import os
import re
import numpy as np
import nltk
import gensim.models.keyedvectors
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import RandomForestClassifier
import joblib

# Globals - start as None
word2vec = None
model_openness = None
model_agreeableness = None
model_neuroticism = None
model_extraversion = None
model_conscientiousness = None
tokenizer_albert = None
model_albert = None
tokenizer_tinybert = None
model_tinybert = None
tokenizer_electra = None
model_electra = None
stop_words = None

app = FastAPI(
    title="Personality Trait Predictor",
    description="Predict Big Five personality traits from a 20-30 word sentence",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_resources():
    global word2vec, model_openness, model_agreeableness, model_neuroticism, model_extraversion, model_conscientiousness
    global tokenizer_albert, model_albert, tokenizer_tinybert, model_tinybert, tokenizer_electra, model_electra
    global stop_words

    if word2vec is not None:
        return

    print("Lazy loading resources on first request...")

    print("Loading NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words("english"))
    print("NLTK and stop words ready.")

    print("Loading GloVe...")
    word2vec = gensim.models.keyedvectors.KeyedVectors.load('models/glove-wiki-gigaword-100.kv')
    print("GloVe loaded.")

    print("Loading models...")
    model_openness = joblib.load("models/model_openness.pkl")
    model_agreeableness = joblib.load("models/model_agreeableness.pkl")
    model_neuroticism = joblib.load("models/model_neuroticism.pkl")
    model_extraversion = joblib.load("models/model_extraversion.pkl")
    model_conscientiousness = joblib.load("models/model_conscientiousness.pkl")
    print("All models loaded.")

    print("Loading transformers...")
    tokenizer_albert = AutoTokenizer.from_pretrained("albert-base-v2")
    model_albert = AutoModel.from_pretrained("albert-base-v2")
    tokenizer_tinybert = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model_tinybert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    tokenizer_electra = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    model_electra = AutoModel.from_pretrained("google/electra-base-discriminator")
    print("Transformers loaded.")

    print("All resources loaded successfully!")

class InputText(BaseModel):
    sentence: str

def preprocess_text(text: str):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def get_post_vector(tokens):
    vectors = [word2vec[word] for word in tokens if word in word2vec]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

def get_embedding(text: str, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

@app.get("/")
def home():
    return {"message": "Personality Predictor API is running! Go to /docs for interactive testing."}

@app.post("/predict")
def predict_traits(data: InputText):
    load_resources()  # Loads everything only once

    sentence = data.sentence.strip()
    tokens = preprocess_text(sentence)
    word_count = len(tokens)

    if not (20 <= word_count <= 30):
        return {"error": f"Sentence must have 20–30 words after cleaning (currently {word_count} words)."}

    vector = get_post_vector(tokens)
    openness = int(model_openness.predict([vector])[0])
    agreeableness = int(model_agreeableness.predict([vector])[0])

    neuro_embedding = get_embedding(sentence, tokenizer_albert, model_albert)
    extra_embedding = get_embedding(sentence, tokenizer_tinybert, model_tinybert)
    consc_embedding = get_embedding(sentence, tokenizer_electra, model_electra)

    neuroticism = int(model_neuroticism.predict([neuro_embedding])[0])
    extraversion = int(model_extraversion.predict([extra_embedding])[0])
    conscientiousness = int(model_conscientiousness.predict([consc_embedding])[0])

    return {
        "Openness": openness,
        "Agreeableness": agreeableness,
        "Neuroticism": neuroticism,
        "Extraversion": extraversion,
        "Conscientiousness": conscientiousness
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
