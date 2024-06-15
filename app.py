import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate a response
def generate_response(input_text, history):
    # Tokenize the input
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    # Append the new user input tokens to the chat history
    bot_input_ids = torch.cat([history, new_user_input_ids], dim=-1) if history is not None else new_user_input_ids
    # Generate a response
    history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(history[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, history

# Custom responses for a friend-like interaction
def friend_response(input_text):
    responses = {
        "hello": ["Hey there! How's it going?", "Hi! What's up?", "Hello! How are you?"],
        "how are you": ["I'm just a bot, but I'm here to chat with you!", "I'm doing great, thanks! How about you?", "I'm always here for you!"],
        "what's your name": ["I'm your friendly chatbot! What's yours?", "I'm just a humble AI, but you can call me Buddy!"],
        "bye": ["Goodbye! It was nice talking to you.", "See you later! Take care.", "Bye! Have a great day!"]
    }

    input_text_lower = input_text.lower()
    for key in responses:
        if key in input_text_lower:
            return random.choice(responses[key])
    return None

# Streamlit user interface
st.title("Your one and only buddy!")
st.write("hey,don't think just talk with me!")

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state.history = None

# User input
input_text = st.text_input("You:", "")

if st.button("Send"):
    if input_text:
        # Check for custom friend-like responses
        friend_reply = friend_response(input_text)
        if friend_reply:
            response = friend_reply
            st.session_state.history = None  # Reset history for next conversation
        else:
            # Generate response from the model
            response, st.session_state.history = generate_response(input_text, st.session_state.history)
        
        # Display the conversation
        st.write(f"You: {input_text}")
        st.write(f"Buddy: {response}")