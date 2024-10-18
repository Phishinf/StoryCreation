import asyncio
from ollama import generate
import chainlit as cl

@cl.on_message
async def on_message(message: cl.Message):
    passing = f"Question {message.content}!"
    response = generate('llama3', passing)
    await cl.Message(response['response']).send()

