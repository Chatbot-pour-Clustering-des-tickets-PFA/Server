from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fuzzywuzzy import process
import joblib


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
    """
    A chatbot function that processes user input, detects intent and emotion, and maintains dialogue state.

    Args:
    - user_input (str): The text input from the user.
    - dialogue_state (dict): Current state of the dialogue.

    Returns:
    - (str, dict): The chatbot's response and the updated dialogue state.
    """
    # Detect intent and emotion using placeholder functions (replace with ML models or APIs)
    intent = detect_intent(user_input)  # Custom function or API
    emotion = detect_emotion(user_input)  # Custom function or API

    # Initialize dialogue state if not provided
    if not dialogue_state:
        dialogue_state = {'history': [], 'last_intent': None, 'emotion': None}

    # Update dialogue state
    dialogue_state['last_intent'] = intent
    dialogue_state['emotion'] = emotion
    dialogue_state['history'].append({'input': user_input, 'intent': intent, 'emotion': emotion})

    # Dynamic responses based on intent and emotion
    if intent == 'greeting':
        response = "Hello! How can I assist you today?"
    elif intent == 'help_request':
        response = "Sure, I'm here to help! Could you tell me more about the issue?"
    elif intent == 'error_report':
        if emotion == 'frustrated':
            response = "I understand this can be frustrating. Can you provide more details about the issue?"
        else:
            response = "Could you share the error code or description so I can assist you better?"
    elif intent == 'goodbye':
        response = "Goodbye! Have a great day!"
    else:
        response = "I'm sorry, I didn't understand that. Could you rephrase?"


    # Return response and updated state
    return response, dialogue_state


def determine_priority_with_emotion(problem, emotion, base_priority_model):
    # Get priority from your model
    base_priority = base_priority_model.predict(problem)

    # Adjust priority based on emotion
    if emotion in ["frustrated", "angry"]:
        adjusted_priority = min(base_priority + 1, 3)  # Increase priority (max = 3)
    elif emotion in ["sad", "worried"]:
        adjusted_priority = base_priority  # Keep base priority
    else:  # neutral or happy
        adjusted_priority = max(base_priority - 1, 1)  # Decrease priority (min = 1)

    return adjusted_priority
