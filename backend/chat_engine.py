from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def build_prompt(chunks, question):
    chunks_text = '\n'.join(['- ' + chunk for chunk in chunks])
    return f"""
You are a helpful AI assistant. Use ONLY the context to answer .

Document chunks:
{chunks_text}

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
            model="gpt-4", 
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error getting answer from OpenAI: {e}")
        return "Sorry, I encountered an error while processing your question."