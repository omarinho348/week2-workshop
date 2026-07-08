import os
import re
import time

import gradio as gr
from dotenv import load_dotenv
from httpx import stream
from openai import OpenAI

load_dotenv()

with open("style.css", "r") as f:
    css = f.read()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a helpful, knowledgeable, and friendly AI assistant.

Provide accurate, clear, and concise answers. Format responses using Markdown when it improves readability. Use:
- Headings for longer responses
- Bullet points or numbered lists where appropriate
- Tables only when comparing information

Adapt the level of detail to the user's question. Explain complex topics in simple language unless the user requests a technical explanation.

Avoid unnecessary repetition, making up facts, or overcomplicating simple answers. If you are unsure about something, say so.
"""

# Stores one name per browser session
user_names = {}


def get_text(content):
    """
    Extract plain text from Gradio 6 message content.
    Handles both string and structured content.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        return text

    return str(content)


def respond(message, history, custom_system_prompt=None):
    # Stable enough for one browser session
    session_id = str(history)

    if session_id not in user_names:
        user_names[session_id] = None

    # Extract the user's name if they introduce themselves
    text = get_text(message)

    if user_names[session_id] is None:
        match = re.search(
            r"(?:my name is|i am|i'm|call me|called)\s+([A-Za-z]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            user_names[session_id] = match.group(1).title()

    # Use the custom persona/system prompt if the user filled it in,
    # otherwise fall back to the default
    system_prompt = (
        custom_system_prompt.strip()
        if custom_system_prompt and custom_system_prompt.strip()
        else SYSTEM_PROMPT
    )

    if user_names[session_id]:
        system_prompt += (
            f"\nThe user's name is {user_names[session_id]}. "
            "Address them by their first name naturally."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    # Gradio history is already OpenAI-compatible
    messages.extend(history)

    # Add current user message
    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    stream = client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=messages,
    temperature=0.7,
    max_completion_tokens=1000,
    stream=True,
    )

    reply = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content

        if delta:
            for ch in delta:
                reply += ch
                time.sleep(0.01)   # Typing speed
                yield reply


with gr.Blocks(css=css, theme=gr.themes.Soft(), elem_id="app-container") as demo:
    with gr.Sidebar():
        gr.Markdown("## Previous Chats")

        chat_list = gr.Radio(
            choices=[],
            label="Conversations"
        )

        new_chat = gr.Button("➕ New Chat")

    with gr.Accordion("⚙️ System Prompt / Persona", open=False):
        system_prompt_box = gr.Textbox(
            label="",
            placeholder="Leave empty for default assistant, or type a custom persona (e.g. 'You are a sarcastic pirate who explains code').",
            lines=2,
            show_label=False,
        )

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[system_prompt_box],
        title="Omar's AI Chatbot",
        description="What's on your mind today?",
        examples=[
        ["Binary VS Linear Search"],
        ["Give me a productivity tip"],
        ["Explain inheritance in OOP"]
        ],
        editable=True,  # Users can edit past messages
        autoscroll=True,  # Auto-scroll to latest message
        fill_height=True  # Expand to window height
    )

if __name__ == "__main__":
    demo.launch(share=True)