import tkinter as tk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging
import os
from accelerate import Accelerator
from torch.cuda.amp import autocast
import openai

# Initialize Accelerator
accelerator = Accelerator()

# Set CUDA_LAUNCH_BLOCKING to get more detailed error reports
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Check if CUDA is available and set the device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# Load model and tokenizer
model_name = "microsoft/Phi-3-mini-4k-instruct"  # Replace with your model name
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the model with Accelerator
model = accelerator.prepare(model)

# Set up text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Configure generation arguments
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.1,
    "do_sample": True,
}

# Initialize conversation history
conversation_history = []

# Define the function to generate chatbot responses
def generate_response(prompt):
    try:
        # Prepare conversation history and current user input
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
        ]

        client = openai.OpenAI(
            api_key="afdec6ff-82f9-425a-8fab-2f487e7bd190",
            base_url="https://api.sambanova.ai/v1",
        )

        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"}],
            temperature =  0.1,
            top_p = 0.1
        )

        print(response.choices[0].message.content)

        # Add previous conversation history
        for idx, user_message in enumerate(conversation_history):
            if idx % 2 == 0:
                messages.append({"role": "user", "content": user_message})
            else:
                messages.append({"role": "assistant", "content": user_message})

        # Add the current user input
        messages.append({"role": "user", "content": prompt})

        # Generate response using the pipeline
        output = pipe(messages, **generation_args)

        # Extract and return the generated response
        response = output[0]['generated_text'].strip()
        return response
    except Exception as e:
        #logger.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error while processing your request."

# Function to handle sending messages
def send_message(event=None):
    user_message = user_input.get()

    if user_message.strip() == "":
        return

    # Generate chatbot response
    response = generate_response(user_message)

    # Add user message and bot response to conversation history
    conversation_history.append(user_message)
    conversation_history.append(response)

    # Display user's message and bot's response in the chatbox
    chat_box.insert(tk.END, f"You:{user_message}\n")
    chat_box.insert(tk.END, f"Assistant:{response}\n")

    # Clear input field
    user_input.delete(0, tk.END)

def clear_chat():
    chat_box.delete(1.0, tk.END)
    conversation_history.clear()

# Create the main window with a modern look
window = tk.Tk()
window.title("Chatbot Assistant")
window.geometry("800x600")
window.configure(bg="#f0f0f0")

# Create a frame for the chat history with padding and modern styling
chat_frame = tk.Frame(window, bg="#f0f0f0")
chat_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Create a scrollbar with a modern look
scrollbar = tk.Scrollbar(chat_frame, orient=tk.VERTICAL)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a text box for displaying chat history with modern styling
chat_box = tk.Text(
    chat_frame, height=30, width=100, yscrollcommand=scrollbar.set,
    bg="white", fg="#333333", font=("Segoe UI", 10), wrap=tk.WORD,
    borderwidth=0, relief=tk.FLAT
)
chat_box.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

scrollbar.config(command=chat_box.yview)

# Create a frame for user input with modern styling
input_frame = tk.Frame(window, bg="#f0f0f0")
input_frame.pack(pady=10, padx=10, fill=tk.X)

# Create an entry widget for user input with modern styling
user_input = tk.Entry(
    input_frame, width=80, font=("Segoe UI", 10),
    bg="#f5f5f5", fg="#333333", relief=tk.FLAT
)
user_input.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True, ipady=8)

user_input.insert(0, "Message:")  # Insert at the start (index 0)

# Function to clear the placeholder text when user focuses on the entry widget
def clear_placeholder(event):
    if user_input.get() == "Message:":
        user_input.delete(0, tk.END)

# Bind the focus-in event to clear the placeholder
user_input.bind("<FocusIn>", clear_placeholder)

# Bind the Return key to send messages
user_input.bind("<Return>", send_message)

# Create a send button with modern styling
send_button = tk.Button(
    input_frame, text="Send", command=send_message,
    bg="#ffcc00", fg="white", font=("Segoe UI", 10, "bold"),
    relief=tk.FLAT, padx=20, pady=10
)
send_button.pack(side=tk.LEFT, padx=10)

# Create a clear chat button with modern styling
clear_button = tk.Button(
    window, text="Clear Chat", command=clear_chat,
    bg="#ffcc00", fg="white", font=("Segoe UI", 10, "bold"),
    relief=tk.FLAT, padx=20, pady=10
)
clear_button.pack(pady=5)

# Run the application
window.mainloop()
