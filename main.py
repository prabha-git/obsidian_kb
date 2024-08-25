import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    StreamlitChatMessageHistory,
)
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

from rag import DocumentRetriever

# Load environment variables from .env file
load_dotenv()

# Constants
SYSTEM_PROMPT = (
    "You are an AI trained to provide detailed responses based on Chat history and Context. "
    "Your answer should be grounded on the context. Say 'I don't know' if no relevant information is found in the context."
)
SESSION_ID_KEY = "session_id"
CHAT_HISTORIES_KEY = "chat_histories"
CHAT_MESSAGES_KEY = "chat_messages"
DEFAULT_LLM_MODEL = "gpt-4o"
ALTERNATE_LLM_MODEL = "llama3"

# Initialize session state
if CHAT_HISTORIES_KEY not in st.session_state:
    st.session_state[CHAT_HISTORIES_KEY] = {}

if SESSION_ID_KEY not in st.session_state:
    st.session_state[SESSION_ID_KEY] = str(uuid.uuid4())

# Initialize chat message history
msgs = StreamlitChatMessageHistory(key=CHAT_MESSAGES_KEY)

# Setup the chat prompt template and chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Question: {question} \n\n Context:\n {context}"),
    ]
)


def get_session_id() -> str:
    """
    Retrieves the current session ID from the session state.

    Returns:
        The current session ID.
    """
    return st.session_state[SESSION_ID_KEY]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Retrieves the chat message history for a given session ID.
    If the session ID doesn't exist in the store, a new ChatMessageHistory is created.

    Args:
        session_id: The session ID to retrieve the history for.

    Returns:
        The chat message history for the given session ID.
    """
    if session_id not in st.session_state[CHAT_HISTORIES_KEY]:
        st.session_state[CHAT_HISTORIES_KEY][session_id] = ChatMessageHistory()
    return st.session_state[CHAT_HISTORIES_KEY][session_id]


def get_session_history_message_texts(session_id: str) -> str:
    """
    Retrieves the chat message texts for a given session ID.

    Args:
        session_id: The session ID to retrieve the message texts for.

    Returns:
        The chat message texts for the given session ID.
    """
    msg_text = ""
    messages = get_session_history(session_id).messages
    for msg in messages:
        msg_text += f"<{msg.type}>: {msg.content}\n"
    return msg_text


# Streamlit UI
st.markdown("""
    <h1 style="
        border: 2px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    ">📔 Personal Knowledge Assistant</h1>
""", unsafe_allow_html=True)

# Remove the sidebar and set a default LLM
llm_choice = "OpenAI"
llm = ChatOpenAI(model=DEFAULT_LLM_MODEL)

runnable = prompt | llm

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

doc_retriever = DocumentRetriever()

# Display and handle chat interaction
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

session_id = get_session_id()

# User input
if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)

    # Retrieve context from documents and add to history
    retrieved_context = doc_retriever.get_relevant_doc(
        user_prompt, chat_history=get_session_history_message_texts(session_id)
    )
    msgs.add_user_message(user_prompt)

    # Process the input using the chain with history
    config = {"configurable": {"session_id": session_id}}

    response = with_message_history.invoke(
        {"question": user_prompt, "context": retrieved_context}, config
    )
    msgs.add_ai_message(response.content)
    st.chat_message("ai").write(response.content)