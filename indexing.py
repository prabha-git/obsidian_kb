import os
from dotenv import load_dotenv
import re
# Load environment variables from .env file
load_dotenv()

from langchain_community.document_loaders import UnstructuredMarkdownLoader

from langchain.indexes import SQLRecordManager, index
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from libs import BQSQLRecordManager
from langchain.indexes import  index


index_name = "obsidian-kb"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = PineconeVectorStore(index=index_name, embedding=embeddings)
rm = BQSQLRecordManager(namespace='obsidian-kb',project_id='obsidian-kb',dataset_id='indexing', table_id='record_manager')
rm.create_schema()
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name,embedding=embeddings,namespace="default")


import re

def clean_content_below_header(content, header="Task Due\n"):
    # Find the position of the header in the content
    header_index = content.find(header)
    if header_index != -1:  # Check if the header exists in the content
        # Keep only the content up to the header
        return content[:header_index]
    return content  # Return the original content if the header is not found


def read_files(directory):
    docs = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                loader = UnstructuredMarkdownLoader(file_path)
                docs.append(loader.load()[0])
                # Cleaning the dataview queries from content
                # clean_content = clean_content_below_header(data[0].page_content)
                # data[0].page_content = f"\n\nFile: {file_path}\n{clean_content}"
                # PineconeVectorStore.from_documents(data, embeddings, index_name=index_name)
                # print(f"path: {file_path} added to pinecone vector database")
    return docs


def index_docs(docs):
    indexing_stats = index(
        docs,
        rm,
        vectorstore,
        cleanup='full',
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )
    return indexing_stats

directory = "../prabha-git.github.io/"
docs = read_files(directory)
indexing_stats = index_docs(docs)
print(f"Indexing stats: {indexing_stats}")

