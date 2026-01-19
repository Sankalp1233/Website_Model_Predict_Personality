print("Starting app startup...")
import os
import re
import numpy as np
import nltk
import gensim.downloader as api
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

print("Downloading NLTK data...")
# Download required NLTK data at startup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
print("NLTK data ready.")

app = FastAPI(
    title="Personality Trait Predictor",
    description="Predict Big Five personality traits from a 20-30 word sentence",
    version="1.0"
)

# Allow your frontend to call this API (change "*" to your actual frontend URL later for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with e.g. "https://your-frontend.onrender.com"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources at startup (this happens once when the app starts)
print("Loading models and embeddings...")

print("Loading stop words...")
stop_words = set(stopwords.words("english"))
print("Stop words loaded.")

# GloVe for Openness & Agreeableness
print("Loading GloVe from file...")
word2vec = gensim.models.keyedvectors.KeyedVectors.load('models/glove-wiki-gigaword-100.kv')
print("GloVe loaded successfully.")

# Load your trained RandomForest models (adjust paths if in a subfolder)
print("Loading model_openness.pkl...")
model_openness = joblib.load("models/model_openness.pkl")
print("model_openness loaded.")

print("Loading model_agreeableness.pkl...")
model_agreeableness = joblib.load("models/model_agreeableness.pkl")
print("model_agreeableness loaded.")

print("Loading model_neuroticism.pkl...")
model_neuroticism = joblib.load("models/model_neuroticism.pkl")
print("model_neuroticism loaded.")

print("Loading model_extraversion.pkl...")
model_extraversion = joblib.load("models/model_extraversion.pkl")
print("model_extraversion loaded.")

print("Loading model_conscientiousness.pkl...")
model_conscientiousness = joblib.load("models/model_conscientiousness.pkl")
print("model_conscientiousness loaded.")

# Transformers
print("Loading ALBERT tokenizer and model...")
tokenizer_albert = AutoTokenizer.from_pretrained("albert-base-v2")
model_albert = AutoModel.from_pretrained("albert-base-v2")
print("ALBERT loaded.")

print("Loading TinyBERT tokenizer and model...")
tokenizer_tinybert = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model_tinybert = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
print("TinyBERT loaded.")

print("Loading ELECTRA tokenizer and model...")
tokenizer_electra = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
model_electra = AutoModel.from_pretrained("google/electra-base-discriminator")
print("ELECTRA loaded.")

print("All models and resources loaded successfully!")


class InputText(BaseModel):
    sentence: str


def preprocess_text(text: str):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())   # remove punctuation & numbers
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
    sentence = data.sentence.strip()
    tokens = preprocess_text(sentence)
    word_count = len(tokens)

    if not (20 <= word_count <= 30):
        return {
            "error": f"Sentence must have 20–30 words after cleaning (currently {word_count} words)."
        }

    # Word2Vec-based features
    vector = get_post_vector(tokens)

    openness = int(model_openness.predict([vector])[0])
    agreeableness = int(model_agreeableness.predict([vector])[0])

    # Transformer-based embeddings
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


# For Render (and local testing)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
