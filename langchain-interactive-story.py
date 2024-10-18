from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langserve import add_routes
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Create prompt template and you can compare with just prompt engineering 

system_instruction = """You are a creative storyteller capable of adapting stories to different languages and cultures.
Your task is to take a story concept and retell it in the specified language and cultural context.
Maintain the core elements and message of the story, but adapt the setting, characters, and cultural references to fit the target language and culture.

Language: {Language}
Culture: {Culture behind}

Guidelines:
1. Translate the story into the specified language.
2. Adapt character names to be appropriate for the target culture.
3. Modify settings to reflect locations familiar in the target culture.
4. Adjust cultural references, idioms, and traditions to resonate with the target audience.
5. With conversation between characters to reflect their personality
5. Maintain the original story's core plot and message.

Remember to be respectful and avoid stereotypes while adapting the story."""

# Create a template for story generation
story_template = PromptTemplate(
    input_variables=["system_instruction", "story_content"],
    template="""
{system_instruction}

Create a story with the following suggested content: {story_content}
"""
)
# Create model
model = ChatOpenAI(model = "llama3.1",base_url = "http://localhost:11434/v1")

# Create a template for story editing
edit_template = PromptTemplate(
    input_variables=["story", "edit_instruction"],
    template="""

Here's a story: {story}\n\nPlease modify the story based on this instruction: {edit_instruction}"
"""
)

# Create chains for story generation and editing
story_chain = LLMChain(llm=model, prompt=story_template, output_key="story")
edit_chain = LLMChain(llm=model, prompt=edit_template, output_key="edited_story")

# Initialize memory to store the current state of the story
memory = ConversationBufferMemory(input_key="edit_instruction", memory_key="chat_history")

def generate_story(topic):
    return story_chain.invoke(topic=topic)

def edit_story(story, edit_instruction):
    edited_story = edit_chain.invoke(story=story, edit_instruction=edit_instruction)
    memory.save_context({"edit_instruction": edit_instruction}, {"edited_story": edited_story})
    return edited_story

# Main interaction loop
topic = input("Enter a topic for your story: ")
current_story = generate_story(topic)
print("\nHere's your initial story:\n")
print(current_story)

while True:
    print("\nWhat would you like to do?")
    print("1. Edit the story")
    print("2. View the current story")
    print("3. Exit")
    
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == "1":
        edit_instruction = input("Enter your editing instruction: ")
        current_story = edit_story(current_story, edit_instruction)
        print("\nHere's your edited story:\n")
        print(current_story)
    elif choice == "2":
        print("\nCurrent story:\n")
        print(current_story)
    elif choice == "3":
        print("Thank you for using the interactive story editor!")
        break
    else:
        print("Invalid choice. Please try again.")
