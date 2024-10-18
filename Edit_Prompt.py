from langchain.prompts import PromptTemplate

# Define the template
template = """
You are a helpful assistant that generates {type_of_content} about {topic}.

Please create a {type_of_content} that:
1. Is appropriate for a {audience} audience
2. Is approximately {length} words long
3. Focuses on the following aspect: {focus}

{additional_instructions}

{type_of_content} Title: {title}

Begin the {type_of_content} below:
"""

# Create the PromptTemplate
prompt = PromptTemplate(
    input_variables=[
        "type_of_content",
        "topic",
        "audience",
        "length",
        "focus",
        "additional_instructions",
        "title"
    ],
    template=template
)
AA	`	```
# Use the prompt
formatted_prompt = prompt.format(
    type_of_content="blog post",
    topic="artificial intelligence",
    audience="general public",
    length="500",
    focus="ethical considerations",
    additional_instructions="Include at least three specific examples.",
    title="The Ethics of AI: Navigating the Future"
)

print(formatted_prompt)