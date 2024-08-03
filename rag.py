from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from libs.custom_multiquery_retriever import CustomMultiQueryRetriever
from libs.custom_selfquery_retriever import CustomSelfQueryRetriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereRerank


class DocumentRetriever:
    metadata_field_info = [
        AttributeInfo(
            name="date",
            description="Date on which journal entry was made (in YYYYMMDD integer format)",
            type="integer",
        ),
    ]
    document_content_description = "Daily Journel that contains the tasks and events of the day"

    def __init__(self, model_name= "text-embedding-3-large", index_name= "obsidian-kb"):
        # Load environment variables
        load_dotenv()

        # Initialize components
        self.llm = ChatOpenAI(model_name='gpt-4o',temperature=0)
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.compressor = CohereRerank(top_n=20)
        self.pinecone_retriever = PineconeVectorStore.from_existing_index(index_name=index_name,namespace='default',
                                                                 embedding=self.embeddings)#.as_retriever(search_kwargs={"k": 20})
        self.retriever=CustomSelfQueryRetriever.from_llm(llm=self.llm,
                                                            vectorstore=self.pinecone_retriever,
                                                            document_contents=self.__class__.document_content_description,
                                                            metadata_field_info=self.__class__.metadata_field_info)

        self.multiquery_retriever_template = PromptTemplate(input_variables=['question','history'],
                                                            template=("""You are an AI language model assistant. Your task is to generate 3 different versions of the given user question by taking previous chat history into account, to retrieve relevant documents from a vector  database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of distance-based similarity search. Provide these alternative questions separated by newlines.
                                                             
Original question: {question} 
                                                              
chat history: {history}
"""))


        self.multiquery_retriever = CustomMultiQueryRetriever.from_llm(self.retriever,llm=self.llm,prompt=self.multiquery_retriever_template)
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor,
                                                                    base_retriever=self.multiquery_retriever)

    def get_relevant_doc(self, query, chat_history):
        retrieved_docs = self.compression_retriever.invoke(input=query, history=chat_history)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return retrieved_context

if __name__=='__main__':
    doc_retreiver = DocumentRetriever()
    hist_msg = ChatMessageHistory()
    hist_msg.add_ai_message("how can  i help")
    print(f""" retrieved documents are {doc_retreiver.get_relevant_doc("What did i do on Mar 9, 2024",hist_msg)}""")