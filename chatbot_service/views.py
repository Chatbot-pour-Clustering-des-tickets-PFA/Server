from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .chatbotFunctions import interactive_chatbot 


@csrf_exempt  
def chatbot_interaction(request):
    if request.method == "POST":
        user_input = request.POST.get('user_input', '')  
        dialogue_state = request.session.get('dialogue_state', {})  

        response, updated_state = interactive_chatbot(user_input, dialogue_state)

        request.session['dialogue_state'] = updated_state

        return JsonResponse({'response': response})

    return JsonResponse({'error': 'Invalid request'}, status=400)


