import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

# -------------------
# 1. Vector store (embeddings only, no chat LLM)
# -------------------
BASE_DIR = Path(__file__).parent
PDF_FOLDER = BASE_DIR / "Books"
VECTORSTORE_PATH = BASE_DIR / "faiss_index"
TOP_K = 3

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Used only to lightly reformat retrieved text (temperature 0 = minimal drift)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_or_load_vectorstore() -> FAISS:
    index_file = VECTORSTORE_PATH / "index.faiss"
    metadata_file = VECTORSTORE_PATH / "index.pkl"

    # Load the saved index if it exists; PDFs are only parsed when it doesn't.
    if index_file.is_file() and metadata_file.is_file():
        return FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))  # rglob("*.pdf") for subfolders
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_FOLDER}")

    docs = []
    for pdf_file in pdf_files:
        docs.extend(PyPDFLoader(str(pdf_file)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(str(VECTORSTORE_PATH))
    return vs


vectorstore = build_or_load_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": TOP_K}
)


# -------------------
# 2. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str  # raw retrieved passages, passed from retrieve -> format


# -------------------
# 3. Nodes
# -------------------
FORMAT_PROMPT = """You are a formatter, not an author.
You receive passages retrieved from documents. Reformat and restructure them
so they are easier to read, with these strict rules:
- Keep the original wording as much as possible; fix only line breaks, broken
  words, stray page numbers/headers and obvious formatting noise.
- Do NOT add facts, explanations, opinions or a conclusion.
- Do NOT remove meaningful content.
- Keep each passage's source label (file name and page) with its text.
- Use light structure only (short headings or bullets) where it helps.
- Do not merge passages that say different things."""


def retrieve_node(state: ChatState):
    query = state["messages"][-1].content
    docs = retriever.invoke(query)

    if not docs:
        return {"context": ""}

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata.get("source", "unknown")).name
        page = doc.metadata.get("page")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""
        parts.append(f"[{i}] {source}{page_label}\n{doc.page_content.strip()}")

    return {"context": "\n\n---\n\n".join(parts)}


def format_node(state: ChatState):
    context = state.get("context", "")
    if not context:
        return {"messages": [AIMessage(content="No relevant passages found.")]}

    query = state["messages"][-1].content
    response = llm.invoke(
        [
            SystemMessage(content=FORMAT_PROMPT),
            HumanMessage(
                content=f"User query (for context only): {query}\n\n"
                        f"Retrieved passages:\n{context}"
            ),
        ]
    )
    return {"messages": [response]}


# -------------------
# 4. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 5. Graph: START -> retrieve -> format -> END
# -------------------
graph = StateGraph(ChatState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("format", format_node)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "format")
graph.add_edge("format", END)

chatbot = graph.compile(checkpointer=checkpointer)


# -------------------
# 6. Helpers
# -------------------
def get_all_threads():
    all_threads = set()
    for chk in checkpointer.list(None):
        all_threads.add(chk.config["configurable"]["thread_id"])
    return list(all_threads)


# Quick test:
# from langchain_core.messages import HumanMessage
# out = chatbot.invoke(
#     {"messages": [HumanMessage(content="your question")]},
#     config={"configurable": {"thread_id": "thread_1"}},
# )
# print(out["messages"][-1].content)