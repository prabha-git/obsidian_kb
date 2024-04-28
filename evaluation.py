from dotenv import load_dotenv
import langsmith
from langsmith import traceable
from rag import DocumentRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain_community.llms import Ollama

from langchain.smith import RunEvalConfig
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    context_relevancy,
    faithfulness,
)

# Load environment variables from .env file
load_dotenv()

client = langsmith.Client()
dataset_name = "obsidian-evaluation"
llm_name='gpt-4-turbo-2024-04-09'
#llm_name='ollama-llama3-8b'
doc_retreiver = DocumentRetriever()

llm = ChatOpenAI(model=llm_name)
#llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI trained to provide detailed responses based on Chat history and Context. Your answer should be grounded on the context. Say 'I don't know' if no relevant information is found in the context."),
        ("human", "Question: {question} \n\n Context:\n {context}"),
    ]
)

runnable = prompt | llm

# Wrap the RAGAS metrics to use in LangChain
evaluators = [
    EvaluatorChain(metric)
    for metric in [
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_entity_recall,
        context_precision,
        context_recall,
        context_relevancy,
        faithfulness,
    ]
]
eval_config = RunEvalConfig(custom_evaluators=evaluators)

def get_rag_context_answer(dataset_dict: dict):
    input = dataset_dict['question']
    retrieved_context = doc_retreiver.get_relevant_doc(input)
    response = runnable.invoke({"question": input, "context": retrieved_context})
    return {
        "answer": response.content,
        "contexts": [retrieved_context],
    }

def evaluate_dataset():
    results = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=get_rag_context_answer,
        evaluation=eval_config,
        project_metadata={'llm_used':llm_name}
)
    return results

# If this is your main module, run the synchronous function like this:
if __name__ == "__main__":
    evaluate_dataset()


