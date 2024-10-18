# This is a code using the local LLM to create the story in different lanaguages with input {content} without Internet
# created by Victor HUNG
# source: https://python.langchain.com/docs/tutorials/llm_chain/#chaining-together-components-with-lcel
# steps: (1) create local environment > python3.11 -m venv ~/LangChain
#        (2) run the local environment > source ~/LangChain/bin/activate
#        (3) go the directory LangChain > cd ~/LangChain
#        (4) create a file directory > mdir chain
#        (5) install necessary packages > pip install -r requirements.txt
#        (6) run the program > python StoryCreation.py
#        (7) browser on http://localhost:8220/chain/playground
#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes
import re

def get_user_input(prompt):
    return input(prompt).strip()

def validate_language_culture(text):
    return bool(re.match(r'^[A-Za-z\s]+$', text))

def get_validated_input(prompt, validator):
    while True:
        user_input = get_user_input(prompt)
        if validator(user_input):
            return user_input
        print("Invalid input. Please try again.")

#1. Create prompt template
system_template = """You are a creative storyteller capable of adapting stories to different languages and cultures.
Your task is to take a story concept and retell it in the specified language and cultural context.
Maintain the core elements and message of the story, but adapt the setting, characters, and cultural references to fit the target language and culture.

Language: {language}
Culture: {culture}

Guidelines:
1. Translate the story into the specified language.
2. Adapt character names to be true in respect to the history.
3. Use the target culture describing the scene of the story.
4. Modify settings to reflect locations, like city or capital, familiar in the target culture.
5. Adjust cultural references, idioms, and traditions to resonate with the target audience.
6. Maintain the original story's core plot and message.

Additional Instructions:
{additional_instructions}

Remember to be respectful and avoid stereotypes while adapting the story."""

prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', 'Create a story with the following suggested content: {content}')
])

def interactive_story_prompt():
    print("Welcome to the Interactive Multilingual Story Generator!")
    
    language = get_validated_input("Enter the target language: ", validate_language_culture)
    culture = get_validated_input("Enter the target culture: ", validate_language_culture)
    content = get_user_input("Enter the story content or theme: ")
    
    print("\nNow, you can add additional instructions for story creation.")
    print("Enter your instructions one by one. Type 'done' when finished.")
    
    additional_instructions = []
    while True:
        instruction = get_user_input("Enter an instruction (or 'done' to finish): ")
        if instruction.lower() == 'done':
            break
        additional_instructions.append(instruction)
    
    formatted_instructions = "\n".join(f"{i+1}. {instr}" for i, instr in enumerate(additional_instructions))
    
    story_request = prompt_template.format(
        language=language,
        culture=culture,
        content=content,
        additional_instructions=formatted_instructions
    )
    
    print("\nGenerated Story Prompt:")
    print(story_request)
    
    return story_request

# 2. Create model
model = ChatOpenAI(model = "llama3.1",base_url = "http://localhost:11434/v1")

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
#chain = prompt_template | model | parser

# You would then pass this `story_prompt` to your LLM to generate the story
story_prompt = interactive_story_prompt()

# 4. App definition
app = FastAPI(
  title="A Local Story Creator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    story_prompt,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8220)


    