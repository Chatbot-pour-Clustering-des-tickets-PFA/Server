from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import process
import joblib
from xgboost import XGBClassifier
from confluent_kafka import Producer

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Label encoder for priorities
base_priority_model = XGBClassifier()
base_priority_model.load_model('best_model_xgboost.json')


producer_config = {
    'bootstrap.servers': 'localhost:9092',  
}
producer = Producer(producer_config)

def send_to_kafka(topic, message):
    producer.produce(topic, value=json.dumps(message))
    producer.flush()


def detect_emotion(user_input):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(user_input)

    if sentiment_scores['compound'] > 0.6:
        return 'happy'
    elif sentiment_scores['compound'] < -0.6:
        return 'angry'
    elif sentiment_scores['compound'] < 0:
        return 'sad'
    else:
        return 'neutral'

    

def detect_intent(user_input):
    """Detects the intent of user input using simple rules."""
    intent_mapping = {
    'greeting': ['hello', 'hi', 'hey', 'good morning', 'good evening'],
    'help_request': ['help', 'assist', 'support', 'how to'],
    'error_report': ['error', 'issue', 'problem', 'bug', 'glitch'],
    'feedback': ['feedback', 'suggestion', 'opinion', 'review'],
    'goodbye': ['bye', 'goodbye', 'see you', 'farewell', 'later'],
    }

    user_input_lower = user_input.lower()
    best_match = None
    best_score = 0

    for intent, keywords in intent_mapping.items():
        match, score = process.extractOne(user_input_lower, keywords)
        if score > best_score:
            best_match = intent
            best_score = score

    return best_match if best_score > 70 else 'unknown'


def interactive_chatbot(user_input, dialogue_state):
    intent = detect_intent(user_input)  # Custom function or API
    emotion = detect_emotion(user_input)  # Custom function or API

    if not dialogue_state:
        dialogue_state = {'history': [], 'last_intent': None, 'emotion': None}

    dialogue_state['last_intent'] = intent
    dialogue_state['emotion'] = emotion
    dialogue_state['history'].append({'input': user_input, 'intent': intent, 'emotion': emotion})

    if intent == 'greeting':
        response = "Hello! How can I assist you today?"
    elif intent == 'help_request':
        response = "Sure, I'm here to help! Could you tell me more about the issue?"
    elif intent == 'error_report':
        # if error report we will submit the ticket
        determine_priority_with_emotion(user_input,emotion)
        if emotion == 'frustrated':
            response = "I understand this can be frustrating. Can you provide more details about the issue?"
        else:
            response = "Could you share the error code or description so I can assist you better?"
    elif intent == 'goodbye':
        response = "Goodbye! Have a great day!"
    else:
        response = "I'm sorry, I didn't understand that. Could you rephrase?"


    return response, dialogue_state





def preprocessing(problem):
    
    vectorized_problem = tfidf_vectorizer.transform([problem])
    
    # Scale the vectorized data
    scaled_problem = scaler.transform(vectorized_problem.toarray())
    
    return scaled_problem



def determine_priority_with_emotion(problem, emotion):
   
    
    preprocessed_problem = preprocessing(problem)
    
    base_priority = base_priority_model.predict(preprocessed_problem)[0]
    
    if emotion in ["frustrated", "angry"]:
        adjusted_priority = min(base_priority + 1, 3)  
    elif emotion in ["sad", "worried"]:
        adjusted_priority = base_priority 
    else: 
        adjusted_priority = max(base_priority - 1, 1)  
    
    
    final_priority = label_encoder.inverse_transform(adjusted_priority)

    message = {
        "problem": problem,
        "emotion": emotion,
        "priority": final_priority
    }

    # Send to Kafka topic
    send_to_kafka("ticketPriority", message)
    return final_priority
