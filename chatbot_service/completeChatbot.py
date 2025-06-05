import os
import pickle
import torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


import torch
from .my_models import Seq2SeqLSTM, Seq2SeqGRU  

import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="")  # Replace with your key

# Load Gemini model (using Gemini 2.0 Flash)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')


EMBED_SIZE    = 128
HIDDEN_SIZE   = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── A) Load your vocabulary mappings ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "vocab.pkl"), "rb") as f:
    vocab_data = pickle.load(f)
stoi = vocab_data["stoi"]
itos = vocab_data["itos"]
new_vocab_size = len(stoi)  # e.g. old_vocab + 2

gru_model = Seq2SeqGRU(new_vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
gru_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "gru_model.pt"),
    map_location=DEVICE
))
gru_model.eval()  # put into eval mode


with open(os.path.join(BASE_DIR, "docs.pkl"), "rb") as f:
    docs = pickle.load(f)

emb_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    os.path.join(BASE_DIR),
    emb_model,
    allow_dangerous_deserialization=True
)




# ─── D) Build your PyTorchSeq2SeqLLM wrapper & RetrievalQA chain ──────────
from pydantic import ConfigDict, Field
from langchain.llms.base import LLM
from langchain.schema      import LLMResult, Generation
from langchain.prompts     import PromptTemplate
from langchain.chains      import RetrievalQA

# If you want a custom “solve only” prompt template, define it here:
solve_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a senior IT support engineer.\n"
        "Context:\n{context}\n\n"
        "User Problem:\n{question}\n\n"
        "Instruction: Provide a concise, step-by-step solution to fix the issue. "
        "Do NOT ask the user for more information—assume you have everything you need.\n\n"
        "Solution:"
    )
)

class PyTorchSeq2SeqLLM(LLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: torch.nn.Module
    stoi:  dict
    itos:  dict
    device: torch.device
    max_len: int

    @property
    def _llm_type(self) -> str:
        return "pytorch-seq2seq"

    def _call(self, prompt: str, stop=None) -> LLMResult:
        # Tokenize the prompt
        tokens = [ self.stoi.get(w, self.stoi['<UNK>']) for w in prompt.split() ]
        src    = torch.tensor(tokens, dtype=torch.long)[None].to(self.device)

        # Initialize decoder with the start token '\t'
        dec_in = torch.tensor([[ self.stoi['\t'] ]], device=self.device)
        gen    = []

        with torch.no_grad():
            for _ in range(self.max_len):
                logits = self.model(src, dec_in)           # [1, t, V]
                next_id = logits[:, -1].argmax(dim=-1).item()
                # If you want to stop on newline, you can uncomment:
                # if next_id == self.stoi.get('\n', -1):
                #     break

                tok = self.itos[next_id]
                gen.append(tok)
                dec_in = torch.cat([ dec_in, torch.tensor([[next_id]], device=self.device) ], dim=1)

        return LLMResult(generations=[[Generation(text=" ".join(gen))]])

# Instantiate the LLM wrapper (we’re using LSTM here; for GRU, just swap):
dl_llm = PyTorchSeq2SeqLLM(
    model=gru_model,
    stoi=stoi,
    itos=itos,
    device=DEVICE,
    max_len=100
)

# Build the RetrievalQA chain. We pass our `solve_template` to override default “ask more” behavior:
qa_chain_dl = RetrievalQA.from_chain_type(
    llm=dl_llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k":5}),
    chain_type_kwargs={"prompt": solve_template}
)



import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from confluent_kafka import Producer
import joblib
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf_vectorizer_priority = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
priority_scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
priority_label_encoder = joblib.load(os.path.join(BASE_DIR, 'label_encoder.pkl'))

intent_vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer_for_detecting_intents.pkl'))
intent_model = joblib.load(os.path.join(BASE_DIR, 'intent_classifier.pkl'))

category_vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer_for_clustering.pkl'))
category_model = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))

priority_model = joblib.load(os.path.join(BASE_DIR, 'best_model_xgboost.pkl'))

producer = Producer({'bootstrap.servers': 'localhost:9092'})
def send_to_kafka(topic, message):
    producer.produce(topic, value=json.dumps(message))
    producer.flush()







def detect_emotion(text):
    scores = SentimentIntensityAnalyzer().polarity_scores(text)
    c = scores["compound"]
    if c > 0.6:   return "happy"
    if c < -0.6:  return "angry"
    if c < 0:     return "sad"
    return "neutral"

def detect_intent(text):
    vect = intent_vectorizer.transform([text])
    intent_prediction = intent_model.predict(vect)
    print("intent: ", intent_prediction)
    return intent_prediction[0]

def predict_priority(text, emotion):
    vect = tfidf_vectorizer_priority.transform([text])
    scaled = priority_scaler.transform(vect.toarray())
    base_priority = priority_model.predict(scaled)[0]

    if emotion in ["angry","frustrated"]:
        adj = min(base_priority + 1, 3)
    elif emotion in ["sad","worried"]:
        adj = base_priority
    else:
        adj = max(base_priority - 1, 1)

    priority_label = priority_label_encoder.inverse_transform([adj])[0]
    return priority_label

def predict_category(text):
    vect = category_vectorizer.transform([text])
    cluster = category_model.predict(vect)[0]
    
    return int(cluster)  # or map it if you have cluster → label mapping


import random

# Dictionnaire centralisé des intents et réponses
intent_responses = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I assist you with?",
        "Hey! How's it going?"
    ],
    "goodbye": [
        "Goodbye! Have a great day!",
        "Bye! Let me know if you need anything else.",
        "See you soon! Take care!"
    ],
    "thank_you": [
        "You're welcome! Let me know if you need anything else.",
        "Happy to help!",
        "No problem at all!"
    ],
    
}

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

import asyncio
import random
import concurrent.futures

executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)


async def interactive_chatbot(user_input, dialogue_state=None, *, user_id:int, model_type):
    if not dialogue_state or "history" not in dialogue_state or "tickets" not in dialogue_state:
        dialogue_state = {
            "history": [],
            "tickets": []
        }

    dialogue_state["history"].append(user_input)

    # Étape 1 : prédire l’intent
    intent = detect_intent(user_input)

    # Étape 2 : si intent est simple → répondre immédiatement sans lancer le traitement lourd
    if intent in intent_responses:
        base_response = random.choice(intent_responses[intent])
        return base_response, dialogue_state,intent

    # Étape 3 : si intent est de type traitement lourd → on déclenche le worker
    elif intent in ["error_report", "feedback","help_request"]:
        loop = asyncio.get_event_loop()
        register_background_task(
            
            user_input, dialogue_state, user_id, model_type, intent
        )
        return "Your Ticket is been invetigated ...", dialogue_state,intent

    # Cas fallback (si intent inconnu)
    return "Thank you for your ticket it is assigned to a technician", dialogue_state, intent

    

def register_background_task(user_input, dialogue_state, user_id, model_type, intent):
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        process_chatbot_logic,
        user_input, dialogue_state, user_id, model_type, intent
    )


def process_chatbot_logic(user_input, dialogue_state, user_id, model_type, intent):
    try:
        # D'abord vérifie
        # r les doublons

        print(" processing intent ", intent)

        if intent in ['greeting', 'thank_you', 'goodbye']:
            print("Intent simple détecté, aucun traitement lourd exécuté.")
            return
        message, dialogue_state = detectDuplicateTickets(user_input, dialogue_state)
        if message == "DUPLICATE_TICKET":
            print("Ticket déjà existant pour cet utilisateur")
            return

        # Puis lancer les modèles
        emotion = detect_emotion(user_input)
        priority = predict_priority(user_input, emotion)
        category = predict_category(user_input)

        ticket_message = {
            "Title": user_input[:50],
            "Description": user_input,
            "Priority": priority,
            "userId": user_id,
            "modelType": model_type,
            "Intent": intent,
            "Emotion": emotion,
            "Category": category
        }

        print("model type", model_type)
        answer = None

        ticket_message["AnswerByDL"] = None

        # Maintenant on déclenche DL et RAG uniquement pour error_report et feedback
        if model_type.lower() == "dl":
            try:
                answer_dl = qa_chain_dl.run(user_input)
                answer = answer_dl.generations[0][0].text
                print("DL answer", answer)
                answer_rag = gemini_rag_answer(user_input)
                ticket_message["AnswerByDL"] = answer
                ticket_message["AnswerByRAG"] = answer_rag
            except Exception as e:
                ticket_message["AnswerByDL"] = f"Erreur DL: {e}"

        elif model_type.lower() == "rag":
            print(" we are in rag")
            try:
                answer = gemini_rag_answer(user_input)
                ticket_message["AnswerByRAG"] = answer
                print("rag answer", answer)
                print(f"RAG answer (repr): {repr(answer)}")
            except Exception as e:
                ticket_message["AnswerByRAG"] = f"Erreur RAG: {e}"
        
        print("priority: ", priority)
        print("category: ",category)

        # Envoi Kafka
        send_to_kafka("ticketCreation", ticket_message)
        print("sent to kafka")
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"user_{user_id}",
            {
                "type": "send_background_result",
                "response": answer  # Ceci sera capté par ChatbotConsumerDLRAG
            }
        )

    except Exception as e:
        print(f"[Erreur globale traitement]: {e}")


from fuzzywuzzy import fuzz


def detectDuplicateTickets(user_input,dialogue_state):
    normalized = user_input.strip().lower()

    # 4b) Exact‐match check against previously recorded tickets
    for prev in dialogue_state["tickets"]:
        if normalized == prev:
            # We already opened a ticket for this exact text in this session
            return (
                "DUPLICATE_TICKET",
                dialogue_state
            )

    # 4c) Fuzzy‐match check against previously recorded tickets
    #     (treat ≥90% similarity as “duplicate”)
    for prev in dialogue_state["tickets"]:
        ratio = fuzz.ratio(normalized, prev)
        if ratio >= 90:
            return (
                "NONE",
                dialogue_state
            )
    dialogue_state["tickets"].append(normalized)  # on enregistre le nouveau ticket
    return ("NEW_TICKET", dialogue_state)
        



def gemini_rag_answer(user_input):
    """
    Use Gemini as a standalone retrieval-augmented reasoning model.
    """
    # Prepare your prompt (you can customize this template)
    prompt = f"""You are an IT support assistant. Provide a technical solution for this user issue:

**User Ticket:** {user_input}

**Solution:** (Be concise, technical, and directly actionable)"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Désolé, une erreur est survenue lors de l'appel à Gemini."
