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

# Download NLTK data at startup (this is fast and safe)
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK data ready.")

app = FastAPI(
    title="Personality Trait Predictor",
    description="Predict Big Five personality traits from a 20-30 word sentence",
    version="1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your GitHub Pages URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables - start as None
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

def load_resources():
    global word2vec, model_openness, model_agreeableness, model_neuroticism, model_extraversion, model_conscientiousness
    global tokenizer_albert, model_albert, tokenizer_tinybert, model_tinybert, tokenizer_electra, model_electra
    global stop_words

    if word2vec is not None:
        print("Resources already loaded.")
        return

    print("Lazy loading resources on first request...")

    print("Loading stop words...")
    stop_words = set(stopwords.words("english"))
    print("Stop words loaded.")

    print("Loading GloVe from file...")
    word2vec = gensim.models.keyedvectors.KeyedVectors.load('models/glove-wiki-gigaword-100.kv')
    print("GloVe loaded successfully.")

    print("Loading models...")
    model_openness = joblib.load("models/model_openness.pkl")
    print("model_openness loaded.")
    model_agreeableness = joblib.load("models/model_agreeableness.pkl")
    print("model_agreeableness loaded.")
    model_neuroticism = joblib.load("models/model_neuroticism.pkl")
    print("model_neuroticism loaded.")
    model_extraversion = joblib.load("models/model_extraversion.pkl")
    print("model_extraversion loaded.")
    model_conscientiousness = joblib.load("models/model_conscientiousness.pkl")
    print("model_conscientiousness loaded.")

    print("Loading ALBERT...")
    tokenizer_albert = AutoTokenizer.from_pretrained("albert-base-v2")
    model_albert = AutoModel.from_pretrained("albert-base-v2")
    print("ALBERT loaded.")

    print("Loading TinyBERT...")
    tokenizer_tinybert = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model_tinybert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    print("TinyBERT loaded.")

    print("Loading ELECTRA...")
    tokenizer_electra = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    model_electra = AutoModel.from_pretrained("google/electra-base-discriminator")
    print("ELECTRA loaded.")

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
    load_resources()  # This runs only once on first request

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

# For local testing (Render ignores this block)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
