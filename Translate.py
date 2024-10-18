# This is a code using the local LLM to do the translate the text without Internet
# created by Victor HUNG
# source: https://python.langchain.com/docs/tutorials/llm_chain/#chaining-together-components-with-lcel
# steps: (1) create local environment > python3.11 -m venv ~/LangChain
#        (2) run the local environment > source ~/LangChain/bin/activate
#        (3) go the directory LangChain > cd ~/LangChain
#        (4) create a file directory > mdir chain
#        (5) install necessary packages > pip install -r requirements.txt
#        (6) at terminal type > export LANGCHAIN_TRACING_V2="true"
#        (7) at terminal type > export LANGCHAIN_API_KEY="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"
#        (8) run the program > python StoryCreation.py
#        (9) browser on http://localhost:8220/chain/playground
#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

# 1. Create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

# 2. Create model
model = ChatOpenAI(model = "llama3.1",base_url = "http://localhost:11434/v1")

# 3. Create parser
parser = StrOutputParser()

# 4. Create chain
chain = prompt_template | model | parser


# 4. App definition
app = FastAPI(
  title="A Local Translator",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8210)


