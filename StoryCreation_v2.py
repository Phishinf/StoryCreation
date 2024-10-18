from pprint import pprint
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langserve import add_routes

# Initial story creation using ChatPromptTemplate
prompt1 = PromptTemplate.from_template(
  'What is the story based on {story content} with {culture} background in {language}. Only respond story.'
)
prompt2 = PromptTemplate.from_template(
  'What is the story? Respond by change according to {instruction}.'
)

# Set up the model and parser
# 0. Create model
chat = ChatOpenAI(model = "llama3.1",base_url = "http://localhost:11434/v1")

#chat = ChatOpenAI(temperature=0.7)
parser = StrOutputParser()

# Create the initial story chain
chain = prompt1 | chat | parser


#def create_initial_story(story_content):
#   return initial_story_chain.invoke({"story_content": story_content})

#def edit_story(story, edit_instruction):
#    return edit_chain.invoke({"story": story, "edit_instruction": edit_instruction})

# combined chain
combined_chain = RunnableSequence(
    {
        "story": chain,
        #"instruction": lambda inputs: inputs['instruction'],
        "instruction": RunnablePassthrough(),
    },
    prompt2,
    chat,
    parser
)

# Iterative editing
# edit_instructions = [
#    "Make the story more suspenseful by adding an unexpected twist.",
#    "Develop the main character's personality more vividly.",
#    "Add more descriptive details about the spell book.",
#]
#    for i, instruction in enumerate(edit_instructions, 1):
#        story = edit_story(story, instruction)
        #print(f"\nEdit {i}:")
        #print(story)
# You can continue this process with more edit instructions as needed
# 4. App definition
app = FastAPI(
  title="A Local Story Creator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    combined_chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8260)

