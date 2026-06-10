from typing import Sequence, TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from app.llm.client import model as chat_model
from app.retrieval.retriever import retrieve_with_scores
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """The Planner state graph.

    This is the first step in the QA pipeline. It takes the user's question and
    produces a retrieval query that will get relevant chunks from the document.
    """

    question: Annotated[Sequence[BaseMessage], add_messages]
    retrieval_query: str
    retrieved_chunks: list
    context_sufficient: bool
    retry_count: int
    answer: str
    next_steps: list[str]





