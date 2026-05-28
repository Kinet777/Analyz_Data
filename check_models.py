import google.generativeai as genai
import os

# Убедитесь, что переменная окружения GEMINI_API_KEY экспортирована
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

print("Доступные модели для генерации текста/мультимодальности:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)