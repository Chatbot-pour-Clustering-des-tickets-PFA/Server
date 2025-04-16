from textblob import TextBlob


def detect_emotion(user_input):
    """Detects emotion based on sentiment analysis."""
    analysis = TextBlob(user_input)
    polarity = analysis.sentiment.polarity

    if polarity > 0.5:
        return 'happy'
    elif polarity < -0.5:
        return 'frustrated'
    elif polarity < 0:
        return 'sad'
    else:
        return 'neutral'
    

def detect_intent(user_input):
    """Detects the intent of user input using simple rules."""
    intent_mapping = {
        'greeting': ['hello', 'hi', 'hey'],
        'help_request': ['help', 'assist', 'support'],
        'error_report': ['error', 'issue', 'problem'],
        'goodbye': ['bye', 'goodbye', 'see you']
    }

    user_input_lower = user_input.lower()

    for intent, keywords in intent_mapping.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return intent

    return 'unknown'  # Default intent if no match found


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
    if emotion == "frustrated":
        response = (
            "I'm sorry to hear that you're frustrated. Can you share more details about what's happening?"
        )
    elif "error" in user_input and "code" not in user_input:
        response = "Could you provide the error code or a screenshot if available? I'll do my best to assist."
    elif intent == "greeting":
        response = "Hello! How can I assist you today?"
    elif intent == "farewell":
        response = "Goodbye! Feel free to reach out anytime."
    else:
        response = "Let me take a closer look and assist you further."

    # Return response and updated state
    return response, dialogue_state