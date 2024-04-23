import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import UnstructuredMarkdownLoader

from langchain.indexes import SQLRecordManager, index
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

index_name = "obsidian-kb"
vectorstore = PineconeVectorStore(index=index_name,embedding=embeddings)

def read_files_and_embed(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                loader = UnstructuredMarkdownLoader(file_path)
                data = loader.load()
                data[0].page_content = f"File: {file_path}\n{data[0].page_content}"
                PineconeVectorStore.from_documents(data, embeddings, index_name=index_name)
                print(f"path: {file_path} added pinecone vector database")

directory = "../prabha-git.github.io/"
read_files_and_embed(directory)