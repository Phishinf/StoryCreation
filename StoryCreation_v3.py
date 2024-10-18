import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory
import chainlit as cl

# Set your OpenAI API key
#os.environ["OPENAI_API_KEY"] = ""
groq_api_key = os.getenv("GROQ_API_KEY")
model_check = 'Llama-3.1-70b-Versatile'
llm = ChatGroq(temperature=0.3, groq_api_key = os.getenv('GROQ_API_KEY'), model_name=model_check)
# Initialize the language model
llm = ChatOpenAI(model = "llama3.1",base_url = "http://localhost:11434/v1")
parser = StrOutputParser()

# Create a conversation memory
memory=ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
    )

# Define the initial story creation prompt
story_prompt = PromptTemplate(
                input_variables=[
                    "culture",
                    "language",
                    "content",
                ],
                template=(
                    """
                    You are a creative storyteller capable of adapting stories to different languages and cultures.
                    Your task is to take a story content and retell it in the specified language and culture context.

                    culture: {culture}
                    language: {language}
                    content: {content} 

                    The story should be approximately 100 words long.
                    Maintain the core elements and message of the story, but adapt the setting, characters
                    Adapt character names to be appropriate for the target culture
                    Modify settings to reflect locations familiar in the target culture
                    Adjust cultural references, idioms, and traditions to resonate with the target audience
                    With conversation between characters to reflect their personality
                    Remember to be respectful and avoid stereotypes while adapting the story."
                    
                    """
                )
            )


# Define the editing prompt
edit_prompt = PromptTemplate(
                input_variables=[
                    "instruction",
                    "story",
                ],
                template=(
                    """
                    Edit the following story based on this instruction.

                    instruction {instruction}
                    story: {story}
                    """
                )
            )

# Create the story creation chain
#story_chain = LLMChain(llm=llm, prompt=story_prompt, memory=memory, output_key="story")
story_chain = story_prompt | llm | parser
#LLMChain(llm=llm, prompt=story_prompt, memory=memory, output_key="story", verbose=False)
#
# Create the editing chain
edit_chain = edit_prompt | llm | parser

@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome to the Story Creation App! Please respond with 'hi, or '好嗎？'").send()
    #await cl.Message("").send()

@cl.on_message
async def main(message: str):
   # Parse the input
    res = await cl.AskUserMessage(content="Input the story features: culture, language, and content, separately with a comma.",timeout=300).send()
    if res:
        await cl.Message(content=f"Your want the story with features: {res['output']}",).send() 
    features = res['output']
    print("The features of the story are " + features)
    inputs = features.split(',')
    #print(inputs)
    if len(inputs) != 3:
        await cl.Message("Please provide input in the format: culture, language, content").send()
        return

    culture, language, content = [input.strip() for input in inputs]

    # Generate the initial story
    #print(culture)
    #print(language)
    #print(content)
    response = await cl.make_async(story_chain.invoke)({"culture": culture, "language": language, "content": content})
    await cl.Message(f"Initial story:\n\n{response}").send()

    # Perform 7 rounds of editing
    story = response
    for i in range(10):
        res = await cl.AskUserMessage(f"Round {i+1}/10: Please provide an editing instruction:", timeout=600).send()
        if res:
            await cl.Message(content=f"Your add the event '{res['output']}' in the story",).send() 
        instruction = res['output']
        print(story)
        print(instruction)
        response = await cl.make_async(edit_chain.invoke)({"instruction": instruction, "story": story})
        await cl.Message(f"Edited story (Round {i+1}):\n\n{response}").send()
        story = response

    await cl.Message("Story creation process completed! Here's your final story:").send()
    await cl.Message(response).send()

if __name__ == "__main__":
    cl.run()
