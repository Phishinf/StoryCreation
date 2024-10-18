from pprint import pprint

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

# prompts
prompt1 = PromptTemplate.from_template(
  'What is the city {person} is from? Only respond with the name of the city.'
)
prompt2 = PromptTemplate.from_template(
  'What country is the city {city} in? Respond in {language}.'
)

# model
model = Ollama(model="llama3.1", temperature=0.0)

# output parser
output_parser = StrOutputParser()

# chain
# chain = prompt1.pipe(model).pipe(output_parser) # This syntax also works
chain = prompt1 | model | output_parser
pprint(chain)

# combined chain
combined_chain = RunnableSequence(
    {
        "city": chain,
        "language": lambda inputs: inputs['language'],
        # "language": RunnablePassthrough(),
    },
    prompt2,
    model,
    output_parser
)

pprint(combined_chain)

result = combined_chain.invoke({
  "person": "習近平",
  "language": "Chinese",
})
print(result)

