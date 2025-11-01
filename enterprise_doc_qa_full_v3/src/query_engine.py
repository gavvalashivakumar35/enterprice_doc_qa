import os
from config.keys import OPENAI_API_KEY if os.path.exists('config/keys.py') else None
def call_openai(prompt):
    try:
        import openai
        openai.api_key = os.getenv('OPENAI_API_KEY') or (OPENAI_API_KEY if 'OPENAI_API_KEY' in globals() else None)
        resp = openai.ChatCompletion.create(model='gpt-4-turbo', messages=[{'role':'user','content':prompt}], max_tokens=300)
        return resp['choices'][0]['message']['content']
    except Exception as e:
        return 'FALLBACK ANSWER: ' + prompt[:400]
def generate_answer(question, contexts):
    prompt = 'Use the following document contexts to answer the question.\n\n' + '\n---\n'.join(contexts) + f"\n\nQuestion: {question}\nAnswer:"
    return call_openai(prompt)
