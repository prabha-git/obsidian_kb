import os
import re
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.indexes import SQLRecordManager, index
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

from libs import BQSQLRecordManager

# Load environment variables from .env file
load_dotenv()

# Constants
INDEX_NAME = "obsidian-kb"
NAMESPACE = "obsidian-kb"
PROJECT_ID = "obsidian-kb"
DATASET_ID = "indexing"
TABLE_ID = "record_manager"
HEADER = "Task Due\n"
DATE_FORMAT = "%Y-%m-%d"
FORCE_UPDATE = (os.environ.get("FORCE_UPDATE") or "false").lower() == "true"

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = PineconeVectorStore(index=INDEX_NAME, embedding=embeddings)
rm = BQSQLRecordManager(
    namespace=NAMESPACE, project_id=PROJECT_ID, dataset_id=DATASET_ID, table_id=TABLE_ID
)
rm.create_schema()
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, embedding=embeddings, namespace="default"
)


def clean_content_below_header(content: str, header: str = HEADER) -> str:
    """
    Clean the content below the specified header.

    Args:
        content: The content to clean.
        header: The header to look for.

    Returns:
        The cleaned content.
    """
    header_index = content.find(header)
    if header_index != -1:
        return content[:header_index]
    return content


def extract_date_from_filename(filename: str) -> Optional[int]:
    """
    Extract the date from the filename.

    Args:
        filename: The filename to extract the date from.

    Returns:
        The extracted date as an integer in the format YYYYMMDD, or None if invalid.
    """
    match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        date_str = match.group(1)
        try:
            date_obj = datetime.strptime(date_str, DATE_FORMAT)
            return int(date_obj.strftime("%Y%m%d"))
        except ValueError:
            return None
    return None


def read_files(directory: str) -> List[UnstructuredMarkdownLoader]:
    """
    Read markdown files from the specified directory.

    Args:
        directory: The directory to read files from.

    Returns:
        A list of loaded documents.
    """
    docs = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                loader = UnstructuredMarkdownLoader(file_path)
                document = loader.load()[0]

                document.page_content = clean_content_below_header(
                    document.page_content
                )
                date = extract_date_from_filename(file)
                document.metadata["date"] = date
                docs.append(document)
    return docs


def index_docs(docs: List[UnstructuredMarkdownLoader]) -> dict:
    """
    Index the documents.

    Args:
        docs: The documents to index.

    Returns:
        The indexing statistics.
    """
    return index(
        docs,
        rm,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        force_update=FORCE_UPDATE,
    )


if __name__ == "__main__":
    directory = "daily_notes_sample"
    docs = read_files(directory)
    indexing_stats = index_docs(docs)
    print(f"Indexing stats: {indexing_stats}")
