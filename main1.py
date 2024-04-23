import streamlit as st
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0)
compressor = CohereRerank()

# Initialize Pinecone retriever
index_name = "obsidian-kb"
pinecone_retriever = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings).as_retriever()

# Initialize Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=pinecone_retriever)


# Initialize chat message history
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Setup the chat prompt template and chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI trained to provide detailed responses based on historical context and document retrieval."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
chain = prompt | ChatOpenAI(model='gpt-4-turbo')
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history"
)

prompt="When did i write about JDBC connection?"
config = {"configurable": {"session_id": "any"}}
response = chain_with_history.invoke({"question": prompt}, config)

print(response)

# Streamlit UI
st.title("Q&A Chat with Enhanced Context and History")

# Display and handle chat interaction
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# User input
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)

    # Retrieve context from documents and add to history
    retrieved_docs = compression_retriever.invoke(prompt)
    retrieved_context = " ".join([doc.page_content for doc in retrieved_docs])  # Adjust this if Document structure is different
    msgs.add_user_message(prompt)
    msgs.add_user_message(retrieved_context)  # Optionally add retrieved docs to history for visibility

    # Process the input using the chain with history
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)

    # Display and add AI response to history
    msgs.add_ai_message(response.content)
    st.chat_message("ai").write(response.content)