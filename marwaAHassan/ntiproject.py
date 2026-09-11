"""
ntiproject.py
--------------
RAG pipeline (Chroma + Gemini + LangGraph + Tavily) for generating a legal
explanation for an invoice-compliance violation, based on Egyptian tax law.

Public interface (this is the ONLY thing the Router should call):

    from marwaAHassan.ntiproject import get_legal_explanation
    result = get_legal_explanation("Invoice missing tax registration number")

Importing this module builds the vectorstore/retriever/LLM/graph once
(module-level setup, same as the original notebook) but does NOT run any
query or print any test output. The old test call now lives under
`if __name__ == "__main__"` so it only runs when you execute this file
directly:

    python ntiproject.py

NOTE: The OCR law-extractor and the synthetic-invoice generator from the
original notebook are intentionally NOT part of this module. They are
one-off, top-level data-prep scripts (they read/write files and pull in
easyocr/pdf2image/reportlab) — bundling them here would make them run
automatically on import, which is exactly what we're avoiding. If the
team still needs them, they should stay as separate standalone scripts
that are run manually / offline, not imported by the Router.
"""

import os
import warnings
from typing import List

from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. API keys — read from environment variables (no Colab, no hardcoding).
#    Set these before importing the module, e.g. in a .env file loaded by
#    the Router, or in your shell / deployment environment:
#
#        export GOOGLE_API_KEY="..."
#        export TAVILY_API_KEY="..."
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Set it as an environment variable "
        "before importing ntiproject."
    )
if not TAVILY_API_KEY:
    raise RuntimeError(
        "TAVILY_API_KEY is not set. Set it as an environment variable "
        "before importing ntiproject."
    )

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# ---------------------------------------------------------------------------
# 2. Vectorstore setup — unchanged from the original notebook logic:
#    load every PDF found in the current working directory, split it, and
#    embed it into an in-memory Chroma collection. No persistence/caching
#    was added here (that would be a change to the RAG logic).
#
#    NOTE: this means the law PDFs must be present in the working directory
#    the Router process is launched from, same as in the original script.
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loader = PyPDFDirectoryLoader(".")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50,
)
doc_splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="legal-invoice-chroma",
    embedding=embeddings,
)

retriever = vectorstore.as_retriever()

# ---------------------------------------------------------------------------
# 3. LLM
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ---------------------------------------------------------------------------
# 4. Grader (decides whether a retrieved doc is relevant)
# ---------------------------------------------------------------------------
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance check on retrieved legal or invoice document, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_grader = """You are a grader assessing whether a retrieved legal or invoice document is relevant to the user query or violation.
If the document contains matching laws, article numbers, or relevant invoice details, grade it as 'yes', otherwise 'no'"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_grader),
    ("human", "Retrieved document: \n\n {document} \n\n User query: {question}"),
])
retrieval_grader = grade_prompt | structured_llm_grader

# ---------------------------------------------------------------------------
# 5. RAG answer chain
# ---------------------------------------------------------------------------
system_rag = """You are a legal and financial assistant. Provide a concise legal opinion using the provided context, mentioning the law and article if available.
If context is insufficient, state that clearly."""

prompt_rag = ChatPromptTemplate.from_messages([
    ("system", system_rag),
    ("human", "Context: {context} \n\n User query: {question} \n\n Provide a concise legal opinion:"),
])
rag_chain = prompt_rag | llm | StrOutputParser()

# ---------------------------------------------------------------------------
# 6. Query rewriter (for web search fallback)
# ---------------------------------------------------------------------------
system_rewriter = """You are an AI assistant that rewrites error messages or legal violation queries into clear queries for web search."""

re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", system_rewriter),
    ("human", "Initial query: \n\n {question} \n Formulate an improved web search query."),
])
question_rewriter = re_write_prompt | llm | StrOutputParser()

web_search_tool = TavilySearchResults(k=3)


# ---------------------------------------------------------------------------
# 7. LangGraph state + nodes
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[Document]


def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search_flag = "No"

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score.lower() == "yes":
            filtered_docs.append(d)
        else:
            web_search_flag = "Yes"

    if not filtered_docs:
        web_search_flag = "Yes"

    return {"documents": filtered_docs, "question": question, "web_search": web_search_flag}


def transform_query(state: GraphState):
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": state["documents"], "question": better_question}


def web_search(state: GraphState):
    question = state["question"]
    documents = state.get("documents", [])

    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results_doc = Document(page_content=web_results)

    documents.append(web_results_doc)
    return {"documents": documents, "question": question}


def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    context = "\n\n".join([d.page_content for d in documents])
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def decide_to_generate(state: GraphState):
    if state["web_search"] == "Yes":
        return "transform_query"
    return "generate"


# ---------------------------------------------------------------------------
# 8. Build + compile the graph (runs once at import time — cheap; no query
#    is executed here, only the graph structure is assembled).
# ---------------------------------------------------------------------------
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# 9. Public interface — this is what the Router calls.
# ---------------------------------------------------------------------------
def get_legal_explanation(query: str) -> str:
    """
    Run the RAG pipeline on `query` and return the generated legal opinion
    as plain text. No extra formatting is applied — the raw pipeline
    output is returned as-is.
    """
    inputs = {"question": query}
    final_output = None
    for output in app.stream(inputs):
        for _key, value in output.items():
            final_output = value
    if final_output and "generation" in final_output:
        return final_output["generation"]
    return "No legal match found."


# ---------------------------------------------------------------------------
# 10. Manual test — only runs when this file is executed directly,
#     NEVER on import (e.g. `from marwaAHassan.ntiproject import ...`).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    error_message = "Invoice missing tax registration number"
    result = get_legal_explanation(error_message)
    print("\n--- RESULT ---")
    print(result)
