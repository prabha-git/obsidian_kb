import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_community.llms import Ollama
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

# Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
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
        ("system", "You are an AI trained to provide detailed responses based on Chat history and Context. Your answer should be grounded on the context. Say 'I don't know' if no relevant information is found in the context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {question} \n\n Context:\n {context}"),
    ]
)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat message history for a given session ID.
    If the session ID doesn't exist in the store, a new ChatMessageHistory is created.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Streamlit UI
st.title("Q&A Chat with Enhanced Context and History")

# Radio button for selecting the LLM, now placed in the sidebar
llm_choice = st.sidebar.radio("Select LLM", ("OpenAI", "Ollama"))

# Dynamic model selection based on user input
if llm_choice == "OpenAI":
    llm = ChatOpenAI(model='gpt-4-turbo')
else:
    llm = Ollama(model="llama3")

runnable = prompt | llm

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Display and handle chat interaction
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# User input
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)

    # Retrieve context from documents and add to history
    retrieved_docs = compression_retriever.invoke(user_prompt)
    retrieved_context = " ".join([doc.page_content for doc in retrieved_docs])
    msgs.add_user_message(user_prompt)

    # Process the input using the chain with history
    config = {"configurable": {"session_id": "any"}}
    response = with_message_history.invoke({"question": user_prompt, "context": retrieved_context}, config)
    if llm_choice == "OpenAI":
        msgs.add_ai_message(response.content)
        # Display and add AI response to history
        st.chat_message("ai").write(response.content)
    else:
        msgs.add_ai_message(response)
        # Display and add AI response to history
        st.chat_message("ai").write(response)

