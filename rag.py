from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereRerank


class DocumentRetriever:
    def __init__(self, model_name= "text-embedding-3-large", index_name= "obsidian-kb"):
        # Load environment variables
        load_dotenv()

        # Initialize components
        self.llm = ChatOpenAI(model_name='gpt-4o',temperature=0)
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.compressor = CohereRerank(top_n=20)
        self.retriever = PineconeVectorStore.from_existing_index(index_name=index_name,namespace='default',
                                                                 embedding=self.embeddings).as_retriever(search_kwargs={"k": 20})
        # self.multiquery_retriever_template = PromptTemplate(input_variables=['question'],
        #                                                     template="""You are an AI language model assistant. Your task is \n
        #                                                     to generate 3 different versions of the given user \n    question by taking previous chat history into account,  to retrieve relevant documents from a vector  database. \n
        #                                                     By generating multiple perspectives on the user question, \n    your goal is to help the user overcome some of the limitations \n
        #                                                     of distance-based similarity search. Provide these alternative \n
        #                                                     questions separated by newlines. Original question: {question}' \n\n
        #                                                     chat history: {history}""")

        self.multiquery_retriever = MultiQueryRetriever.from_llm(self.retriever,llm=self.llm)
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor,
                                                                    base_retriever=self.multiquery_retriever)

    def get_relevant_doc(self, query):
        retrieved_docs = self.compression_retriever.invoke(query)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return retrieved_context

if __name__=='__main__':
    doc_retreiver = DocumentRetriever()
    print(f""" retrieved documents are {doc_retreiver.get_relevant_doc("What did i do on Mar 9, 2024")}""")