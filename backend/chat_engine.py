from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(chunks, question):
    return f"""
You are a helpful AI assistant. Use ONLY the below document chunks to answer the question.

Document chunks:
{''.join(['- ' + chunk + '\n' for chunk in chunks])}

Question: {question}

Instructions:
- Answer based only on the provided document chunks
- If the answer is not found in the chunks, say: "I don't know based on the provided documents."
- Be specific and cite relevant information from the chunks
- Keep your answer concise but comprehensive
"""

def get_answer(chunks, question):
    prompt = build_prompt(chunks, question)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for cheaper option
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error getting answer from OpenAI: {e}")
        return "Sorry, I encountered an error while processing your question."