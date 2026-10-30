from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import sqlite3
import requests
import os
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path

load_dotenv()

# print("ALPHA_API_KEY:", os.getenv('ALPHA_API_KEY'))

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------
# 1B. RAG Settings 
# -------------------
pdf_folder = Path(__file__).parent / "Books"
pdf_files = sorted(pdf_folder.glob("*.pdf"))  # Use rglob("*.pdf") to include subfolders.

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found in {pdf_folder}")

docs = []
for pdf_file in pdf_files:
    docs.extend(PyPDFLoader(str(pdf_file)).load())


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

vectorstore_path = Path(__file__).parent / "faiss_index"
index_file = vectorstore_path / "index.faiss"
metadata_file = vectorstore_path / "index.pkl"

if index_file.is_file() and metadata_file.is_file():
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(vectorstore_path))



# -------------------
# 2. Tools
# -------------------
# Tools
# search_tool = DuckDuckGoSearchRun(region="us-en")

# @tool
# def calculator(first_num: float, second_num: float, operation: str) -> dict:
#     """
#     Perform a basic arithmetic operation on two numbers.
#     Supported operations: add, sub, mul, div
#     """
#     try:
#         if operation == "add":
#             result = first_num + second_num
#         elif operation == "sub":
#             result = first_num - second_num
#         elif operation == "mul":
#             result = first_num * second_num
#         elif operation == "div":
#             if second_num == 0:
#                 return {"error": "Division by zero is not allowed"}
#             result = first_num / second_num
#         else:
#             return {"error": f"Unsupported operation '{operation}'"}
        
#         return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
#     except Exception as e:
#         return {"error": str(e)}


# @tool
# def get_stock_price(symbol: str) -> dict:
#     """
#     Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
#     using Alpha Vantage with API key in the URL.
#     """
#     url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('ALPHA_API_KEY')}"
#     r = requests.get(url)
#     return r.json()

@tool
def rag_tool(query: str) -> str:
    """
    A tool that uses a vectorstore to retrieve relevant documents based on a query.
    """
    results = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrieved_docs = results.get_relevant_documents(query)
    context = [doc.page_content for doc in retrieved_docs]
    metadata = [doc.metadata for doc in retrieved_docs]
    return{
        
        "query": query,
        "context": context,
        "metadata": metadata
    }


# tools = [search_tool, get_stock_price, calculator]
tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
# -------------------
# 5. Checkpointer
# -------------------
# Checkpointer
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')


chatbot = graph.compile(checkpointer=checkpointer)


## For testing the chatbot

# response = chatbot.invoke({'messages': [HumanMessage(content="what is history of this capital in 50 words ?")]} , config={'configurable': {'thread_id': 'thread_1'}})
# print(response['messages'][-1].content)

# print("Threading is working fine")

# state = chatbot.get_state(config={'configurable': {'thread_id': 'thread_1'}})
# print(state.values.get('messages')[-1].content)

# for chk in checkpointer.list(None):
#     print(chk.config)

## Retrieve all the threads from the checkpointer
def get_all_threads():
    all_threads = set()
    for chk in checkpointer.list(None):
        all_threads.add(chk.config['configurable']['thread_id'])
    return list(all_threads)