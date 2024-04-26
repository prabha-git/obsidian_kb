from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereRerank


class DocumentRetriever:
    def __init__(self, model_name= "text-embedding-3-large", index_name= "obsidian-kb"):
        # Load environment variables
        load_dotenv()

        # Initialize components
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.compressor = CohereRerank()
        self.retriever = PineconeVectorStore.from_existing_index(index_name=index_name,
                                                                 embedding=self.embeddings).as_retriever()
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor,
                                                                    base_retriever=self.retriever)

    def get_relevant_doc(self, query):
        retrieved_docs = self.compression_retriever.invoke(query)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return retrieved_context

if __name__=='__main__':
    doc_retreiver = DocumentRetriever()
    print(doc_retreiver.get_relevant_doc("What did i do on Mar 9, 2024"))