import streamlit as st
import json
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")
API_KEY_2 = os.getenv("API_KEY_2")
API_BASE_2 = os.getenv("API_BASE_2")
MODEL_AGGREGATE = os.getenv("MODEL_AGGREGATE")
MODEL_REFERENCE_1 = os.getenv("MODEL_REFERENCE_1")
MODEL_REFERENCE_2 = os.getenv("MODEL_REFERENCE_2")
MODEL_REFERENCE_3 = os.getenv("MODEL_REFERENCE_3")

# Constants
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
ROUNDS = int(os.getenv("ROUNDS", 1))

def generate_with_references(model, messages, references, temperature, max_tokens, api_base, api_key):
    # Implement the logic from the original code
    system_message = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:
"""
    for i, reference in enumerate(references):
        system_message += f"\n{i+1}. {reference}"

    messages = [{"role": "system", "content": system_message}] + messages

    response = requests.post(
        f"{api_base}",
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature if temperature > 1e-4 else 0,
            "max_tokens": max_tokens,
            "stream": True
        },
        headers={
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        },
        stream=True
    )
    return response

def process_chunk(chunk):
    if 'choices' in chunk and chunk['choices']:
        for choice in chunk['choices']:
            if 'delta' in choice and 'content' in choice['delta']:
                return choice['delta']['content']
    return ""

def main():
    st.set_page_config(page_title="LLM Chat App", page_icon="🤖", layout="wide")
    st.title("LLM Chat Application")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        st.text(f"{message['role']}: {message['content']}")

    # React to user input
    prompt = st.text_input("What is your question?")
    if st.button("Send"):
        if prompt:
            # Display user message
            st.text(f"user: {prompt}")
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Generate responses from reference models
            reference_models = [MODEL_REFERENCE_1, MODEL_REFERENCE_2, MODEL_REFERENCE_3]
            references = []

            for model in reference_models:
                response = generate_with_references(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    references=[],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    api_base=API_BASE,
                    api_key=API_KEY
                )
                reference_output = ""
                for line in response.iter_lines():
                    if line:
                        line_decoded = line.decode('utf-8').strip()
                        if line_decoded.startswith('data: {'):
                            json_data = line_decoded[6:]
                            try:
                                chunk = json.loads(json_data)
                                reference_output += process_chunk(chunk)
                            except json.JSONDecodeError:
                                pass
                references.append(reference_output)

            # Generate final response using the aggregate model
            final_response = generate_with_references(
                model=MODEL_AGGREGATE,
                messages=[{"role": "user", "content": prompt}],
                references=references,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                api_base=API_BASE_2,
                api_key=API_KEY_2
            )

            # Display assistant response
            full_response = ""
            message_placeholder = st.empty()
            for line in final_response.iter_lines():
                if line:
                    line_decoded = line.decode('utf-8').strip()
                    if line_decoded.startswith('data: {'):
                        json_data = line_decoded[6:]
                        try:
                            chunk = json.loads(json_data)
                            content = process_chunk(chunk)
                            full_response += content
                            message_placeholder.text(f"assistant: {full_response}")
                        except json.JSONDecodeError:
                            pass
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()