import os
os.environ["OPENAI_API_KEY"]=""

import json
from langchain_community.document_loaders import DirectoryLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Load documents
loader = PyPDFLoader("")
documents = loader.load()

# Add filename metadata to documents
for document in documents:
    document.metadata['filename'] = document.metadata['source']

# Initialize generator with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# Adapt to Korean
language = "korean"
generator.adapt(language, evolutions=[simple, reasoning,conditional,multi_context])
generator.save(evolutions=[simple, reasoning, multi_context,conditional])

# Generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, is_async=False)

# Save
print(testset)
new=testset
addtestset=new.to_pandas()
csv_filename="dataset.csv"
addtestset.to_csv(csv_filename)

