# Install necessary libraries if not already installed
try:
    import transformers
    import torch
    import gradio as gr
except ImportError:
    !pip install transformers torch gradio

from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Load model and tokenizer
model_name = "ibm-granite/granite-3.3-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define response generation function
def generate_response(user_input):
    messages = [{"role": "user", "content": user_input}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Create Gradio interface
interface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    title="Citizen AI - Real-Time Conversational AI Assistant",
    description="Ask me anything about the Citizen AI project."
)

# Launch Gradio app (inline for Colab)
interface.launch(share=True)
