"""
Production Financial Research Agent API
Cleaned & Refactored Version
"""

import os
from typing import Literal, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

# =========================================================
# SIMPLE LOGGER (replace with real logging if needed)
# =========================================================
class Logger:
    def info(self, msg):
        print(msg)

logger = Logger()

# =========================================================
# DATA LOADING + VECTOR STORE
# =========================================================

def load_vectorstore():
    folder = "data/Companies-AI-Initiatives"

    if not os.path.exists(folder):
        logger.info("⚠️ Dataset folder missing. RAG disabled.")
        return None

    loader = PyPDFDirectoryLoader(folder)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    return Chroma.from_documents(
        docs,
        embeddings,
        collection_name="AI_Initiatives"
    )

VECTORSTORE = None


def get_vectorstore():
    global VECTORSTORE
    if VECTORSTORE is None:
        VECTORSTORE = load_vectorstore()
    return VECTORSTORE


# =========================================================
# RAG TOOL
# =========================================================

@tool
def query_private_database(query: str) -> str:
    """Search internal AI initiative reports."""

    try:
        vs = get_vectorstore()
        if not vs:
            return "Private database unavailable."

        retriever = vs.as_retriever(search_kwargs={"k": 10})
        docs = retriever.get_relevant_documents(query)

        if not docs:
            return "No relevant information found."

        context = "\n".join(d.page_content for d in docs)

        prompt = f"""
You are an analyst AI assistant.

Answer ONLY using the context below.
If not found, say: "I don't know."

Context:
{context}

Question:
{query}
"""

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        return model.invoke(prompt).content

    except Exception as e:
        return f"Database error: {str(e)}"


# =========================================================
# AGENT SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an autonomous Financial Research Analyst.

Always:
- analyze financials
- analyze sentiment
- check AI initiatives
- identify risks
- give recommendation

Be structured.
Cite sources.
"""


# =========================================================
# DUMMY TOOL STUBS (replace with real ones)
# =========================================================

@tool
def get_stock_price(ticker: str) -> str:
    return f"Stock price lookup placeholder for {ticker}"

@tool
def get_stock_history(ticker: str, period: str) -> str:
    return f"History placeholder {ticker} {period}"

@tool
def search_financial_news(query: str) -> str:
    return f"News placeholder for {query}"

@tool
def analyze_sentiment(text: str) -> str:
    return "Sentiment neutral (placeholder)"


# =========================================================
# AGENT CREATION
# =========================================================

def create_agent():

    tools = [
        get_stock_price,
        get_stock_history,
        search_financial_news,
        analyze_sentiment,
        query_private_database
    ]

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools(tools)

    def agent_node(state):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def route(state) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(dict)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        route,
        {"tools": "tools", "end": END},
    )

    graph.add_edge("tools", "agent")

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# =========================================================
# LAZY LOAD AGENT (CLOUD SAFE)
# =========================================================

_AGENT = None

def get_agent():
    global _AGENT
    if _AGENT is None:
        logger.info("Initializing agent...")
        _AGENT = create_agent()
    return _AGENT


# =========================================================
# API ENTRYPOINT
# =========================================================

def run_agent(payload: Dict[str, Any]) -> Dict[str, str]:
    query = (payload or {}).get("query", "").strip()

    if not query:
        return {"error": "Missing query"}

    agent = get_agent()

    config = {
        "configurable": {
            "thread_id": payload.get("thread_id", "default")
        }
    }

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )

    return {"answer": result["messages"][-1].content}


# =========================================================
# LOCAL TEST
# =========================================================

if __name__ == "__main__":
    print(
        run_agent(
            {"query": "Analyze NVDA and its AI strategy"}
        )
    )
