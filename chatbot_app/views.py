# chatbot_app/views.py
import os
import sys
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain import OpenAI
from decouple import Config, RepositoryEnv

# # Load the index only once, outside the view, for better performance
# index = GPTSimpleVectorIndex.load_from_disk('index.json')

# Load OpenAI API key from the environment variable file
config = Config(RepositoryEnv(".env"))
OPENAI_API_KEY = config('OPENAI_API_KEY')
# print("OpenAI API Key:", OPENAI_API_KEY)

# Configure the OpenAI API key
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

@csrf_exempt
def chatbot_api(request):
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
           
            question = data.get('question')
            
            if question:
                index = GPTSimpleVectorIndex.load_from_disk('index.json')
                # print('index',index)
                response = index.query(question)
                # print('response',response.response)
                return JsonResponse({'response': response.response})
            else:
                return JsonResponse({'error': 'Invalid input.'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed.'}, status=405)
