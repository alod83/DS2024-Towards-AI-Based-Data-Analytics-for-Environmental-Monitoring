from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import json
import os


loader = DirectoryLoader('output/')
data = loader.load()

#loader = PyPDFDirectoryLoader('output/')
#data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 20,
    length_function = len,
    is_separator_regex= False
)

splitted_data = text_splitter.split_documents(data)

with open('secrets.json') as f:
    secrets = json.load(f)

secrets = secrets['gpt-4']
#embeddings = AzureOpenAIEmbeddings(
#    azure_endpoint = secrets['endpoint'],
#    api_key        = secrets['api_key'],
#    api_version    = "2023-05-15",
#    model='gpt-4',
    #deployment_name='text-embedding-ada-002',
    #azure_deployment=secrets['deployment_name']
#)

embeddings = OpenAIEmbeddings()
embedding_function = OpenAIEmbeddingFunction(
    api_key =os.environ.get('OPENAI_API_KEY'),
    model_name = 'text-embedding-ada-002'
)

store = Chroma.from_documents(
    splitted_data,
    embeddings,
    ids = [f"{item.metadata['source']}-{index}" for index, item in enumerate(splitted_data)],
    persist_directory = 'db' 
)

store.persist()

prompt = PromptTemplate(
    template = """Consider {context} where:
    - Year is the year
    - Month is the month
    - Mean is the average monthly temperature
    - Max is the maximum monthly temperature
    - Min is the minimum monthly temperature
    - Mode is the most frequent monthly temperature value

    Answer the following question: {question}""",
    input_variables=['context', 'question'],
)

llm = AzureChatOpenAI(
    azure_endpoint = secrets['endpoint'],
    api_key        = secrets['api_key'],
    api_version    = "2023-05-15",
    deployment_name=secrets['deployment_name'],
    #model_name=secrets['model_name']
)

llm_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = store.as_retriever(),
    chain_type_kwargs = {'prompt' : prompt},
    return_source_documents = False
)

#question = "In which month and year was the highest temperature value recorded?"
#question = "What was the hottest month for each year?"
question = "What is the percentage increase in temperature over the decade 2013-2023?"

print(llm_chain(question))

#response = llm([message])

#response = llm.invoke("Translate this sentence from English to French. I love programming.")    

#response = llm.invoke("Tell me a joke")

#response.choices[0].message.content

#print(response)