import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory, StreamlitChatMessageHistory
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from rag import DocumentRetriever

import uuid


# Load environment variables from .env file
load_dotenv()

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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


def get_session_id():
    return st.session_state.session_id

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat message history for a given session ID.
    If the session ID doesn't exist in the store, a new ChatMessageHistory is created.
    """
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

def get_session_history_message_texts(session_id: str) -> str:

    msg_text=""
    messages = get_session_history(session_id).messages
    for msg in messages:
        msg_text+="<"+msg.type+">"+": "+msg.content+"\n"
    return msg_text

# Streamlit UI
st.title("Q&A Chat with Enhanced Context and History")

# Radio button for selecting the LLM, now placed in the sidebar
llm_choice = st.sidebar.radio("Select LLM", ("OpenAI", "Ollama"))

# Dynamic model selection based on user input
if llm_choice == "OpenAI":
    llm = ChatOpenAI(model='gpt-4o')
else:
    llm = Ollama(model="llama3")

runnable = prompt | llm

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

doc_retreiver = DocumentRetriever()

# Display and handle chat interaction
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)



session_id = get_session_id()
# User input
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)

    # Retrieve context from documents and add to history
    print(st.session_state.chat_histories)
    retrieved_context = doc_retreiver.get_relevant_doc(user_prompt, chat_history=get_session_history_message_texts(session_id))
    msgs.add_user_message(user_prompt)

    # Process the input using the chain with history
    config = {"configurable": {"session_id": session_id}}

    response = with_message_history.invoke({"question": user_prompt, "context": retrieved_context}, config)
    if llm_choice == "OpenAI":
        msgs.add_ai_message(response.content)
        # Display and add AI response to history
        st.chat_message("ai").write(response.content)
    else:
        msgs.add_ai_message(response)
        # Display and add AI response to history
        st.chat_message("ai").write(response)

