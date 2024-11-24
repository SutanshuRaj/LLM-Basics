import os
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.runnables import RunnableSequence
from pprint import pprint
# from langchain_community.llms import Ollama

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage )

load_dotenv(find_dotenv())

display = False
output_parser = StrOutputParser()


### Models: LLM Wrappers
model = ChatOpenAI(model='gpt-4o', temperature=0.3)
# model = Ollama(model="gemma:7b", temperature=0.0)

prompt = [SystemMessage(content='You are a Data Scientist.'),
    HumanMessage(content='Write pseudo-code for linear regression.')]

if display:
    output = model.invoke(prompt)
    print(output.content, end='\n')


### Prompt Templates
template = """ You are a Data Scientist.
            Explain the concept of {concept} in 2-3 lines."""

prompt_template = PromptTemplate.from_template(template=template)

if display:
    # prompt_output = model.invoke(prompt_template.format(concept='auto-encoders'))
    prompt_output = model.invoke(prompt_template.format(concept='regularization'))
    print(prompt_output.content, end='\n')


### Chains
chain = prompt_template | model | output_parser

# chain = prompt_template.pipe(model).pipe(output_parser)
# chain = LLMChain(llm=model, prompt=prompt_template)
# pprint(chain)

# result = chain.invoke({"concept": "auto-encoders"})
# print(result)

template_02 = """ You are a Data Scientist.
            Explain the concept of {concept} in {number} words."""

prompt_template_02 = PromptTemplate.from_template(template=template_02)

# from langchain.chains import SimpleSequentialChain
chain_exp = RunnableSequence(
    {
        'concept' : chain,
        'number' : lambda input : input['number']
    },
    prompt_template_02,
    model,
    output_parser)

if True:
    result_exp = chain_exp.invoke({'concept': 'decision trees', 'number': 500})
    # print(result_exp)


### Embeddings and VectorStore.
text_idx = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=0)
text_chunk = text_idx.create_documents([result_exp])
# print('Quick sanity check: ', text_chunk[0].page_content)

embedding_matrix = OpenAIEmbeddings(model='text-embedding-ada-002')

# query_vector = embedding_matrix.embed_query(text_chunk[0].page_content)

pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = 'langchain-quickstart'
# index_name = pc.Index('langchain-quickstart')
vectors_text = PineconeVectorStore.from_documents(text_chunk,
                        embedding=embedding_matrix,
                        index_name=index_name)


### Query the Database.
query = 'What are the main features of decision trees?'
result = vectors_text.similarity_search(query)
print(result)


### Agents
