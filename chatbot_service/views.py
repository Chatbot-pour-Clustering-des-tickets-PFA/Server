from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chatbotFunctions import interactive_chatbot 


@csrf_exempt  
def chatbot_interaction(request):
    if request.method == "POST":
        user_input = request.POST.get('user_input', '')  
        dialogue_state = request.session.get('dialogue_state', {})  

        # Call the chatbot function to process the input
        response, updated_state = interactive_chatbot(user_input, dialogue_state)

        # Save the updated state to the session
        request.session['dialogue_state'] = updated_state

        # Return the response as JSON
        return JsonResponse({'response': response})

    return JsonResponse({'error': 'Invalid request'}, status=400)


