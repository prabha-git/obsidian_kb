from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


index_name = "obsidian-kb"
llm = Cohere(temperature=0)
compressor = CohereRerank()

pinecone_retriever = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings).as_retriever()

query = "Find out JDBC connection string"


compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=pinecone_retriever
)

compressed_docs = compression_retriever.invoke(
   query
)

print(compressed_docs)
