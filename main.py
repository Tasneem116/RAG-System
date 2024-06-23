# Import necessary modules for using OpenAI, prompts, Elasticsearch, and text processing
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Define a prompt template for question-answering
template = '''
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use five sentences minimum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
'''

# Load documents from a text file
loader = TextLoader(r"C:\Users\Dell\Desktop\Resume Projects\RAG Project\text.txt")
documents = loader.load()

# Split the loaded documents into smaller chunks of 500 characters each
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create embeddings for the document chunks using OpenAI
embeddings = OpenAIEmbeddings()

# # Initialize an ElasticsearchStore to store the document chunks and their embeddings
# vector_db = ElasticsearchStore.from_documents(
#     docs,  # Document chunks to store
#     embedding=embeddings,  # Embeddings for the documents
#     index_name='rag_project1',  # Name of the Elasticsearch index
#     es_cloud_id="For_RAG_Project:dXMtZWFzdC0yLmF3cy5lbGFzdGljLWNsb3VkLmNvbTo0NDMkYjBmYTMwY2I2ZTlmNDBiY2JkMzJjOTc3MzEwMGE3ZTgkNzEwMzQ0ZjAzMTg3NGE5ZjhhOTQ4ODNhMGRiNmRlMWY=",  # Cloud ID for Elasticsearch
#     es_api_key="SWpxSVJwQUI1cTNQRUZIWjVCcEw6VXR1MkJFczdUc09zYmdOWjRNSFdYdw=="  # API key for Elasticsearch
# )


# Initialize an ElasticsearchStore to store the document chunks and their embeddings
vector_db = ElasticsearchStore(
    docs,  # The document chunks to be stored in Elasticsearch
    embedding=embeddings,  # The embeddings for the document chunks
    index_name='rag_project1',  # The name of the Elasticsearch index to store the documents
    es_cloud_id="For_RAG_Project:dXMtZWFzdC0yLmF3cy5lbGFzdGljLWNsb3VkLmNvbTo0NDMkYjBmYTMwY2I2ZTlmNDBiY2JkMzJjOTc3MzEwMGE3ZTgkNzEwMzQ0ZjAzMTg3NGE5ZjhhOTQ4ODNhMGRiNmRlMWY=",  # Cloud ID for the Elasticsearch cluster
    es_api_key="SWpxSVJwQUI1cTNQRUZIWjVCcEw6VXR1MkJFczdUc09zYmdOWjRNSFdYdw=="  # API key for authenticating with the Elasticsearch cluster
)


# Create a retriever from the Elasticsearch store to fetch relevant document chunks
retriever = vector_db.as_retriever()

# Create a prompt template object from the defined template
prompt = ChatPromptTemplate.from_template(template)

# Initialize the GPT-4 model with zero temperature (deterministic output)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define a Retrieval-Augmented Generation (RAG) chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # Define the input schema for the chain
    | prompt  # Use the prompt template to format the input
    | llm  # Pass the formatted input to the LLM for generating answers
    | StrOutputParser()  # Parse the output as a string
)

# Define a query to get information about King John's family
query = "given some family information on king john"

# Get the response from the RAG chain by invoking the query
response = rag_chain.invoke(query)

# Print the response
print(response)
