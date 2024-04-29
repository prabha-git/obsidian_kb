import os
from dotenv import load_dotenv
import re
# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import UnstructuredMarkdownLoader

from langchain.indexes import SQLRecordManager, index
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

index_name = "obsidian-kb"
vectorstore = PineconeVectorStore(index=index_name, embedding=embeddings)

import re

def clean_content_below_header(content, header="Task Due\n"):
    # Find the position of the header in the content
    header_index = content.find(header)
    if header_index != -1:  # Check if the header exists in the content
        # Keep only the content up to the header
        return content[:header_index]
    return content  # Return the original content if the header is not found


def read_files_and_embed(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                loader = UnstructuredMarkdownLoader(file_path)
                data = loader.load()
                # Cleaning the dataview queries from content
                clean_content = clean_content_below_header(data[0].page_content)
                data[0].page_content = f"\n\nFile: {file_path}\n{clean_content}"
                PineconeVectorStore.from_documents(data, embeddings, index_name=index_name)
                print(f"path: {file_path} added to pinecone vector database")

directory = "../prabha-git.github.io/"
read_files_and_embed(directory)
