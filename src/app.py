import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

if os.environ.get("AZURE_OPENAI_API_KEY") is None:
    raise ValueError("Please set the azure_openai_deployment environment variable")

from llm.graph import stream_event


def chat(message, history):
    return stream_event(message)


gr.ChatInterface(fn=chat, type="messages").launch(server_port=9000)
