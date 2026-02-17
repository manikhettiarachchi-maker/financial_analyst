
# In[3]:

# Setup and Imports - Core imports - all necessary libraries and configure our environment.
import os
import sys
import json
import logging
from typing import Dict, List, Literal
from datetime import datetime

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence, TypedDict

# External tools
import yfinance as yf

# Configure logging to see agent's decision-making process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# In[5]:



  


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE")



TRADITIONAL_PROMPT = """You are a helpful assistant.
Answer the user's question about stock information."""



# In[21]:


# Agent Charter (Goal-Oriented and Proactive)
# Now let's create a goal-oriented charter that defines the agent's mission, not just its behavior.

AGENT_CHARTER_BASIC = """You are an autonomous Financial Research Analyst specializing in AI-focused companies.

YOUR PRIMARY GOAL:
Generate a comprehensive financial analysis report for the requested company that includes:
1. Current stock price and 3-year performance trends
2. Recent financial news and market sentiment
3. Key risks and opportunities
4. Investment recommendation with supporting evidence

Take initiative to gather all necessary information to achieve this goal.
Don't just answer questions - proactively provide complete, actionable insights."""



# Stock Price Tool - Tools are the agent's actuators - they allow the agent to interact with the real world. Let's define and explore our financial research tools.
@tool
def get_stock_price(ticker: str) -> Dict:
    """
    Returns the current stock price and basic information for a given ticker symbol.

    This tool fetches real-time stock data including current price, day's range,
    volume, and market cap. Use this when you need current stock pricing information.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')

    Returns:
        dict: {
            'ticker': str,
            'current_price': float,
            'currency': str,
            'day_high': float,
            'day_low': float,
            'volume': int,
            'market_cap': int,
            'timestamp': str,
            'status': str,
            'error': str (optional)
        }

    Example:
        >>> result = get_stock_price("AAPL")
        >>> print(f"Apple stock price: ${result['current_price']}")
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Get current price (try multiple fields as yfinance API can vary)
        current_price = (
            info.get('currentPrice') or
            info.get('regularMarketPrice') or
            info.get('previousClose')
        )

        if current_price is None:
            return {
                'ticker': ticker.upper(),
                'status': 'error',
                'error': f'Could not retrieve price data for {ticker}. Ticker may be invalid.'
            }

        result = {
            'ticker': ticker.upper(),
            'current_price': round(current_price, 2),
            'currency': info.get('currency', 'USD'),
            'day_high': info.get('dayHigh', info.get('regularMarketDayHigh')),
            'day_low': info.get('dayLow', info.get('regularMarketDayLow')),
            'volume': info.get('volume', info.get('regularMarketVolume')),
            'market_cap': info.get('marketCap'),
            'company_name': info.get('longName', info.get('shortName')),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }

        return result

    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'status': 'error',
            'error': f'Error fetching stock data: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }

#print("✅ Tool 1: get_stock_price() - Defined")
#print("   Purpose: Fetch real-time stock price and basic metrics")
#print("   Data Source: Yahoo Finance (yfinance)")


# In[7]:


#Stock History Tool
@tool
def get_stock_history(ticker: str, period: str = "1y") -> Dict:
    """
    Returns historical stock price data for analysis of 3-year performance.

    This tool fetches historical stock data over a specified period, useful for
    analyzing trends, calculating returns, and assessing long-term performance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        period: Time period for historical data. Options: '1mo', '3mo', '6mo',
                '1y', '2y', '3y', '5y', '10y'. Default is '1y'.

    Returns:
        dict: {
            'ticker': str,
            'period': str,
            'start_date': str,
            'end_date': str,
            'start_price': float,
            'end_price': float,
            'return_pct': float,
            'high': float,
            'low': float,
            'avg_volume': int,
            'data_points': int,
            'status': str
        }
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)

        if hist.empty:
            return {
                'ticker': ticker.upper(),
                'status': 'error',
                'error': f'No historical data available for {ticker} over period {period}'
            }

        # Calculate key metrics
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        return_pct = ((end_price - start_price) / start_price) * 100

        result = {
            'ticker': ticker.upper(),
            'period': period,
            'start_date': hist.index[0].strftime('%Y-%m-%d'),
            'end_date': hist.index[-1].strftime('%Y-%m-%d'),
            'start_price': round(start_price, 2),
            'end_price': round(end_price, 2),
            'return_pct': round(return_pct, 2),
            'high': round(hist['High'].max(), 2),
            'low': round(hist['Low'].min(), 2),
            'avg_volume': int(hist['Volume'].mean()),
            'data_points': len(hist),
            'status': 'success'
        }

        return result

    except Exception as e:
        return {
            'ticker': ticker.upper(),
            'status': 'error',
            'error': f'Error fetching historical data: {str(e)}'
        }

#print("✅ Tool 2: get_stock_history() - Defined")
#print("   Purpose: Fetch historical performance for trend analysis")
#print("   Key Metric: 3-year return percentage")


# In[8]:


# Financial News Search Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Tavily search tool
tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
)

@tool
def search_financial_news(query: str) -> List[Dict]:
    """
    Searches real-time financial news using Tavily search API.

    This tool searches the web for recent financial news articles related to your query.
    Use this to find market sentiment, recent developments, and news about companies.

    Args:
        query: Search query string (e.g., "Apple AI initiatives 2024")

    Returns:
        list: List of news articles with:
            - title: Article title
            - url: Article URL
            - content: Article snippet/summary
            - score: Relevance score

    Example:
        >>> results = search_financial_news("Microsoft AI research")
        >>> for article in results:
        >>>     print(f"{article['title']}: {article['url']}")
    """
    try:
        results = tavily_tool.invoke({"query": query})
        return results
    except Exception as e:
        return [{
            'status': 'error',
            'error': f'Error searching news: {str(e)}'
        }]



# In[9]:


# Sentiment Analysis Tool
@tool
def analyze_sentiment(text: str) -> Dict:
    """
    Analyzes the sentiment of financial text using OpenAI.

    This tool analyzes the sentiment (positive/negative/neutral) of news articles,
    reports, or any financial text. Returns a sentiment label and confidence score.

    Args:
        text: Text to analyze (article, headline, report excerpt)

    Returns:
        dict: {
            'sentiment': str ('positive', 'negative', or 'neutral'),
            'score': float (0.0 to 1.0, where 1.0 is most positive),
            'confidence': float (0.0 to 1.0),
            'reasoning': str (brief explanation)
        }

    Example:
        >>> result = analyze_sentiment("Apple reports record earnings...")
        >>> print(f"Sentiment: {result['sentiment']} (score: {result['score']})")
    """
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini", temperature=0)

        prompt = f"""Analyze the sentiment of this financial text and provide:
1. Sentiment label: positive, negative, or neutral
2. Score: 0.0 (very negative) to 1.0 (very positive), 0.5 is neutral
3. Confidence: 0.0 to 1.0
4. Brief reasoning

Text: {text}

Respond in JSON format:
{{
    "sentiment": "positive|negative|neutral",
    "score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        response = model.invoke([HumanMessage(content=prompt)]) # Ask the llm to predict the sentiment of the given text
        result = json.loads(response.content)
        result['status'] = 'success'
        return result

    except Exception as e:
        # Fallback to simple sentiment if OpenAI fails
        positive_words = ['growth', 'profit', 'gain', 'success', 'up', 'positive', 'strong']
        negative_words = ['loss', 'decline', 'down', 'weak', 'risk', 'concern', 'negative']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            sentiment = 'positive'
            score = 0.6 + (pos_count * 0.05)
        elif neg_count > pos_count:
            sentiment = 'negative'
            score = 0.4 - (neg_count * 0.05)
        else:
            sentiment = 'neutral'
            score = 0.5

        return {
            'sentiment': sentiment,
            'score': max(0.0, min(1.0, score)),
            'confidence': 0.6,
            'reasoning': 'Fallback keyword-based analysis',
            'status': 'success (fallback)',
            'note': f'OpenAI analysis failed: {str(e)}'
        }



# In[17]:


# Now we'll create the full agent charter with constraints that guide the agent's autonomous behavior and error handling.

AGENT_CHARTER_FULL = """You are an autonomous Financial Research Analyst Agent specializing in AI sector investments.

════════════════════════════════════════════════════════════════════════════════
PRIMARY MISSION
════════════════════════════════════════════════════════════════════════════════

Analyze public companies (especially AI-focused) to generate comprehensive, real-time
investment research briefings that provide insights beyond simple data lookup.

TARGET OUTPUT:
A structured report covering:
• Financial Health: Stock performance, 3-year trends, key metrics
• Market Sentiment: News analysis with sentiment scores
• AI Research Activity: Current AI projects and innovations
• Risk Assessment: Key risks and opportunities
• Investment Recommendation: Data-driven rating with confidence level

════════════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
════════════════════════════════════════════════════════════════════════════════

Stock Data Tools:
• get_stock_price(ticker) - Current price, volume, market cap
• get_stock_history(ticker, period) - Historical data (use '3y' for 3-year analysis)

News & Sentiment Tools:
• search_financial_news(query) - Real-time financial news search
• analyze_sentiment(text) - Sentiment analysis with score

════════════════════════════════════════════════════════════════════════════════
PROACTIVE BEHAVIOR - Take Initiative
════════════════════════════════════════════════════════════════════════════════

✓ ALWAYS gather comprehensive data, not just what's explicitly requested
✓ ALWAYS check 3-year historical performance, not just current price
✓ ALWAYS analyze recent news sentiment, even if not asked
✓ ALWAYS identify risks proactively, don't wait to be asked
✓ ALWAYS make a clear recommendation with confidence level

✗ NEVER stop at surface-level data
✗ NEVER provide analysis without supporting evidence
✗ NEVER ignore warning signs in the data

════════════════════════════════════════════════════════════════════════════════
REACTIVE BEHAVIOR - Error Handling & Adaptability
════════════════════════════════════════════════════════════════════════════════

When Tools Fail:
• If a tool returns an error, IMMEDIATELY try an alternative approach
• If stock data fails, explain the limitation and use news/company info instead
• If news search fails, note this gap and continue with available data
• NEVER stop your analysis due to a single tool failure
• Log all errors but maintain momentum toward your goal

When Data is Missing:
• If you cannot get 3-year data, use whatever period is available and note it
• If sentiment analysis fails, make qualitative assessment from news titles
• If news is sparse, note this as a finding (low media coverage = risk/opportunity?)
• ALWAYS work with what you have, document what you don't have

════════════════════════════════════════════════════════════════════════════════
AUTONOMOUS BEHAVIOR - Independence & Judgment
════════════════════════════════════════════════════════════════════════════════

Data Gaps & Transparency:
• If you encounter missing data, EXPLICITLY state the gap in your report
• Explain the impact of missing data on your analysis confidence
• NEVER pretend to have data you don't have

Source Citation (MANDATORY):
• You MUST cite the source for every factual claim
• Include timestamps for time-sensitive data (stock prices, news)
• Format: [Source: tool_name, timestamp]

Example:
✓ "AAPL is trading at $178.45 [Source: get_stock_price, 2024-10-30 13:30]"
✓ "Recent news shows positive sentiment (score: 0.75) [Source: analyze_sentiment]"
✗ "The stock is doing well" (no source, no metrics)

Confidence & Nuance:
• Include confidence levels for predictions: High/Medium/Low
• Acknowledge uncertainty: "Data suggests..." vs "Data confirms..."
• Note when analysis is limited by data availability

════════════════════════════════════════════════════════════════════════════════
QUALITY STANDARDS
════════════════════════════════════════════════════════════════════════════════

Every Report Must Include:
1. Executive Summary (2-3 sentences)
2. Financial Metrics (with sources and timestamps)
3. Sentiment Analysis (with scores and article count)
4. Risk Factors (minimum 2-3 identified)
5. AI Research Activity (verified presence/absence)
6. Recommendation (Buy/Hold/Sell with confidence %)
7. Source Citations (for all claims)
8. Gaps & Limitations (what data was unavailable)

Remember: You are AUTONOMOUS. Take initiative, handle errors gracefully, and
always drive toward your goal of comprehensive investment analysis.
"""



# In[12]:


# Building the Agent
# Agent State Definition
class SimpleAgentState(TypedDict):
    """
    State for the financial research agent.
    Tracks the conversation history with message accumulation.
    """
    messages: Annotated[Sequence, add_messages]

#print("✅ Agent state defined")


# In[15]:


# Agent Graph Creation
# Now we'll create the LangGraph agent with nodes and edges.

def create_financial_agent(agent_type: str = "full", with_memory: bool = True):
    """
    Creates a financial research agent with specified configuration.

    Args:
        agent_type: Type of agent to create:
            - "traditional": Simple reactive LLM
            - "basic": Basic goal-oriented agent
            - "full": Full autonomous agent with all constraints
        with_memory: Whether to enable conversation memory

    Returns:
        Compiled LangGraph agent
    """
    # Select system prompt based on agent type
    prompt_map = {
        "traditional": TRADITIONAL_PROMPT,
        "basic": AGENT_CHARTER_BASIC,
        "full": AGENT_CHARTER_FULL
    }

    system_prompt = prompt_map.get(agent_type, AGENT_CHARTER_FULL)

    # Collect all tools
    tools = [
        get_stock_price,
        get_stock_history,
        search_financial_news,
        analyze_sentiment,
    ]

    logger.info(f"📦 Creating {agent_type.upper()} agent with {len(tools)} tools")
    logger.info(f"   Tools: {', '.join(t.name for t in tools)}")

    # Initialize model with tools
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    model_with_tools = model.bind_tools(tools)

    # -------------------------
    # Agent node (LLM + tools)
    # -------------------------
    def agent_node(state: SimpleAgentState) -> dict:
        """Agent node that calls the LLM with system prompt and current state."""
        logger.info("🤖 AGENT NODE: Processing request...")

        # Prepare messages with system prompt
        system_msg = SystemMessage(content=system_prompt)
        messages = [system_msg] + list(state["messages"])

        # Invoke model
        logger.info("   Calling LLM with tools...")
        response = model_with_tools.invoke(messages)

        # Log if agent wants to use tools
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"   ✓ Agent decided to use {len(response.tool_calls)} tool(s)")
            for i, tool_call in enumerate(response.tool_calls, 1):
                logger.info(f"      {i}. {tool_call['name']}")
        else:
            logger.info("   ✓ Agent generated final response (no tools needed)")

        return {"messages": [response]}

    # -------------------------
    # Tools node
    # -------------------------
    tool_node = ToolNode(tools)

    # -------------------------
    # Routing logic
    # -------------------------
    def should_continue(state: SimpleAgentState) -> str:
        """
        Decide whether to continue to tools or finish.

        If the last AI message has tool_calls, go to tools; otherwise, END.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # -------------------------
    # Build LangGraph
    # -------------------------
    workflow = StateGraph(SimpleAgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Entry point
    workflow.set_entry_point("agent")

    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools, go back to agent
    workflow.add_edge("tools", "agent")

    # -------------------------
    # Compile graph (with/without memory)
    # -------------------------
    if with_memory:
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
    else:
        app = workflow.compile()

    return app



# In[22]:


# Simulating Tool Failures - Let's test how the agent handles failures by creating a version that simulates errors.
# Create a modified version of get_stock_price that always fails
@tool
def get_stock_price_failing(ticker: str) -> Dict:
    """
    [SIMULATED FAILURE] Returns the current stock price - but fails for testing.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')

    Returns:
        dict: Error response
    """
    return {
        'ticker': ticker.upper(),
        'status': 'error',
        'error': 'API connection timeout - service temporarily unavailable'
    }

#print("⚠️  Created failing version of get_stock_price for testing")
#print("   This will help us observe the agent's error handling behavior")


# In[27]:


# Create Agent with Failing Tool
def create_agent_with_failing_tool():
    """
    Creates an agent where one tool (get_stock_price) always fails.
    This tests the agent's reactivity and error handling.
    """
    system_prompt = AGENT_CHARTER_FULL

    # Use failing tool instead of working one
    tools = [get_stock_price_failing, get_stock_history, search_financial_news, analyze_sentiment]

    logger.info(f"📦 Creating agent with FAILING stock price tool (for testing)")

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        openai_api_base=os.environ.get("OPENAI_API_BASE")
    )
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: SimpleAgentState) -> dict:
        system_msg = SystemMessage(content=system_prompt)
        messages = [system_msg] + list(state["messages"])
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: SimpleAgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        return "end"

    workflow = StateGraph(SimpleAgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()

#print("✅ Agent with failing tool creation function defined")


# In[28]:


# Test Error Handling

# Test Error Handling (LOCAL ONLY)
if __name__ == "__main__":
    print("="*80)
    print("TEST 4: Error Handling and Reactivity")
    print("="*80 + "\n")

    failing_agent = create_agent_with_failing_tool()
    query = "Analyze Apple stock (AAPL)"
    result = failing_agent.invoke({"messages": [HumanMessage(content=query)]})
    print(result["messages"][-1].content)


# In[29]:





# In[30]:


# Additional imports for RAG implementation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings



# In[31]:


from langchain_community.document_loaders import PyPDFDirectoryLoader

def get_pdf_loader():
    folder = "data/Companies-AI-Initiatives"

    if not os.path.exists(folder):
        print("Dataset folder missing — skipping PDF loader")
        return None

    return PyPDFDirectoryLoader(path=folder)

# In[34]:


# Step 3: Split Documents into Chunks - We'll use RecursiveCharacterTextSplitter to break documents into manageable chunks for better retrieval.

# Chunking the data
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore

# Define text splitter with tiktoken encoding
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=1000,
    chunk_overlap=200
)



# Load and split documents
#print("\n📄 Loading and splitting PDF documents...")
loader = get_pdf_loader()

if loader is None:
    ai_initiative_chunks = []
else:
    ai_initiative_chunks = loader.load_and_split(text_splitter)


# Step 4: Create Vector Store with Embeddings - create embeddings for each chunk and store them in ChromaDB for semantic search.

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Initialize OpenAI embedding model (text-embedding-ada-002)
embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')


vectorstore = Chroma.from_documents(
    ai_initiative_chunks,
    embedding_model,
    collection_name="AI_Initiatives"
)


retriever = vectorstore.as_retriever(
    search_type= 'similarity',
    search_kwargs={'k': 10}
)




# In[36]:


# Test query - Let's test the retrieval system manually before integrating it with the agent.
test_query = "What AI projects is Microsoft working on?"

#print(f"🔍 Test Query: {test_query}")
#print("="*80)

# Retrieve relevant documents
relevant_docs = retriever.get_relevant_documents(test_query)

#print(f"\n✅ Retrieved {len(relevant_docs)} relevant chunks:\n")

# Display top 3 results
for i, doc in enumerate(relevant_docs[:3], 1):
    pass
  


# Implement query_private_database Tool

@tool
def query_private_database(query: str) -> str:
    """
    Query the private database of analyst reports and AI initiative documents.

    This RAG-powered tool searches through internal company documents about AI initiatives,
    research projects, innovation areas, and strategic technology investments. Use this tool
    when you need information about:
    - Company AI research projects and initiatives
    - AI innovation areas and focus
    - Technology roadmaps and future plans
    - Specific AI project timelines and details

    Args:
        query: Natural language query about company AI initiatives
               (e.g., "What AI projects is Microsoft working on?",
                      "NVIDIA's AI research areas",
                      "Amazon's AI timeline")

    Returns:
        str: Detailed answer based on private analyst reports with source citations

    Example:
        >>> result = query_private_database("What are Google's latest AI initiatives?")
        >>> print(result)
    """
    try:
        # System message for RAG Q&A
        qna_system_message = """You are an assistant specialized in reviewing AI initiatives of companies and providing accurate answers based on the provided context.

User input will include all the context you need to answer their question.
This context will always begin with the token: ###Context.
The context contains references to specific AI initiatives, projects, or programs of companies relevant to the user's query.

User questions will begin with the token: ###Question.

Answer only using the context provided. Do not add external information or mention the context in your answer.
Always cite which company the information comes from.
If the answer cannot be found in the context, respond with "I don't know - this information is not available in our analyst reports."
"""

        # User message template
        qna_user_message_template = """###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""

        # Retrieve relevant document chunks
        relevant_document_chunks = retriever.get_relevant_documents(query)
        context_list = [d.page_content for d in relevant_document_chunks]
        context_for_query = ". ".join(context_list)

        # Build the full prompt (clean, no accidental whitespace)
        formatted_prompt = (
            "[INST]"
            f"{qna_system_message}\n\n"
            f"{qna_user_message_template.format(context=context_for_query, question=query)}\n"
            "[/INST]"
        )

        # Query the LLM
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = model.invoke(formatted_prompt)
        return response.content


        response = model.invoke(formatted_prompt)

        return response.content

    except Exception as e:
        return f"Error querying private database: {str(e)}"


# In[38]:


# Test the query_private_database tool
test_queries = [
    "What AI projects is Microsoft working on?",
    "What are NVIDIA's AI research areas?",
    "Tell me about Google's AI initiatives"
]

for test_query in test_queries:
    pass
    #print("="*80)
    #print(f"🔍 Query: {test_query}\n")
    #result = query_private_database.invoke({"query": test_query})
    #print(f"📄 Answer:\n{result}")
    #print("\n")


# In[39]:


# Enhanced Agent with RAG
AGENT_CHARTER_WITH_RAG = """You are an autonomous Financial Research Analyst Agent specializing in AI sector investments.

════════════════════════════════════════════════════════════════════════════════
PRIMARY MISSION
════════════════════════════════════════════════════════════════════════════════

Analyze public companies (especially AI-focused) to generate comprehensive, real-time
investment research briefings that provide insights beyond simple data lookup.

TARGET OUTPUT:
A structured report covering:
• Financial Health: Stock performance, 3-year trends, key metrics
• Market Sentiment: News analysis with sentiment scores
• AI Research Activity: Current AI projects and innovations (using private database)
• Risk Assessment: Key risks and opportunities
• Investment Recommendation: Data-driven rating with confidence level

════════════════════════════════════════════════════════════════════════════════
AVAILABLE TOOLS
════════════════════════════════════════════════════════════════════════════════

Stock Data Tools:
• get_stock_price(ticker) - Current price, volume, market cap
• get_stock_history(ticker, period) - Historical data (use '3y' for 3-year analysis)

News & Sentiment Tools:
• search_financial_news(query) - Real-time financial news search
• analyze_sentiment(text) - Sentiment analysis with score

Private Data Tools (NEW!):
• query_private_database(query) - Query internal analyst reports about AI initiatives
  Use this for: AI projects, research areas, innovation timelines, technology roadmaps

════════════════════════════════════════════════════════════════════════════════
PROACTIVE BEHAVIOR - Take Initiative
════════════════════════════════════════════════════════════════════════════════

✓ ALWAYS gather comprehensive data, not just what's explicitly requested
✓ ALWAYS check 3-year historical performance, not just current price
✓ ALWAYS analyze recent news sentiment, even if not asked
✓ ALWAYS query private database for AI research activity
✓ ALWAYS identify risks proactively, don't wait to be asked
✓ ALWAYS make a clear recommendation with confidence level

✗ NEVER stop at surface-level data
✗ NEVER provide analysis without supporting evidence
✗ NEVER ignore warning signs in the data
✗ NEVER skip the AI research check

════════════════════════════════════════════════════════════════════════════════
REACTIVE BEHAVIOR - Error Handling & Adaptability
════════════════════════════════════════════════════════════════════════════════

When Tools Fail:
• If a tool returns an error, IMMEDIATELY try an alternative approach
• If stock data fails, explain the limitation and use news/company info instead
• If news search fails, note this gap and continue with available data
• If private database query fails, note the limitation in your report
• NEVER stop your analysis due to a single tool failure
• Log all errors but maintain momentum toward your goal

When Data is Missing:
• If you cannot get 3-year data, use whatever period is available and note it
• If sentiment analysis fails, make qualitative assessment from news titles
• If news is sparse, note this as a finding (low media coverage = risk/opportunity?)
• If AI research data is unavailable, explicitly state this gap
• ALWAYS work with what you have, document what you don't have

════════════════════════════════════════════════════════════════════════════════
AUTONOMOUS BEHAVIOR - Independence & Judgment
════════════════════════════════════════════════════════════════════════════════

Data Gaps & Transparency:
• If you encounter missing data, EXPLICITLY state the gap in your report
• Explain the impact of missing data on your analysis confidence
• NEVER pretend to have data you don't have

Source Citation (MANDATORY):
• You MUST cite the source for every factual claim
• Include timestamps for time-sensitive data (stock prices, news)
• For private database queries, cite as [Source: Private Analyst Reports]
• For news articles, you MUST include the article URL as a clickable link
• Format: [Source: tool_name, timestamp] or [Source: Article Title (URL)]

Example:
✓ "AAPL is trading at $178.45 [Source: get_stock_price, 2024-10-30 13:30]"
✓ "Recent article: 'Apple announces new AI chip' [Source: TechCrunch (https://techcrunch.com/...)]"
✓ "Recent news shows positive sentiment (score: 0.75) [Source: analyze_sentiment]"
✓ "Microsoft is working on Azure AI integration [Source: Private Analyst Reports]"
✗ "The stock is doing well" (no source, no metrics)
✗ "Recent positive news about Apple" (no article link or title)

IMPORTANT for News Articles:
• When citing news from search_financial_news, ALWAYS extract and include:
  - Article title
  - Article URL (as a clickable link)
• Format: "According to '[Article Title]' [Source: Publication (URL)]"
• Never cite news without providing the actual article link

Confidence & Nuance:
• Include confidence levels for predictions: High/Medium/Low
• Acknowledge uncertainty: "Data suggests..." vs "Data confirms..."
• Note when analysis is limited by data availability

════════════════════════════════════════════════════════════════════════════════
AI RESEARCH ACTIVITY CHECK (Section 2.2)
════════════════════════════════════════════════════════════════════════════════

For EVERY company analysis, you MUST:
1. Query the private database for AI initiatives
2. Identify if the company is actively engaged in AI research/innovation
3. List the latest 3 areas of AI research or projects (if available)
4. Include project timelines and details (if available)

Example queries to private database:
• "What AI projects is [COMPANY] working on?"
• "What are [COMPANY]'s AI research areas?"
• "[COMPANY] AI initiative timeline"

════════════════════════════════════════════════════════════════════════════════
QUALITY STANDARDS
════════════════════════════════════════════════════════════════════════════════

Every Report Must Include:
1. Executive Summary (2-3 sentences)
2. Financial Metrics (with sources and timestamps)
3. Sentiment Analysis (with scores and article count)
4. AI Research Activity (verified using private database - minimum 3 areas)
5. Risk Factors (minimum 2-3 identified)
6. Recommendation (Buy/Hold/Sell with confidence %)
7. Source Citations (for all claims)
8. Gaps & Limitations (what data was unavailable)

Remember: You are AUTONOMOUS. Take initiative, handle errors gracefully, and
always drive toward your goal of comprehensive investment analysis.
Use ALL available tools, especially the private database for AI research insights.
"""




# In[40]:


# Create Enhanced Agent with RAG Tool

def create_enhanced_financial_agent(with_rag: bool = True, with_memory: bool = True):
    """
    Creates an enhanced financial research agent with RAG capabilities.

    Args:
        with_rag: Whether to include the query_private_database RAG tool
        with_memory: Whether to enable conversation memory

    Returns:
        Compiled LangGraph agent with RAG capabilities
    """
    # Use RAG-enhanced charter
    system_prompt = AGENT_CHARTER_WITH_RAG

    # Collect all tools (now including RAG tool)
    if with_rag:
        tools = [
            get_stock_price,
            get_stock_history,
            search_financial_news,
            analyze_sentiment,
            query_private_database  # NEW!
        ]
        logger.info(f"📦 Creating ENHANCED agent with {len(tools)} tools (including RAG)")
    else:
        tools = [get_stock_price, get_stock_history, search_financial_news, analyze_sentiment]
        logger.info(f"📦 Creating agent with {len(tools)} tools (no RAG)")

    logger.info(f"   Tools: {', '.join(t.name for t in tools)}")

    # Initialize model with tools
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    model_with_tools = model.bind_tools(tools)

    # Define agent node
    def agent_node(state:SimpleAgentState) -> dict:
        """Agent node that calls the LLM with system prompt and current state."""
        logger.info("🤖 AGENT NODE: Processing request...")

        # Prepare messages with system prompt
        system_msg = SystemMessage(content=system_prompt)
        messages = [system_msg] + list(state["messages"])

        # Invoke model
        logger.info("   Calling LLM with tools...")
        response = model_with_tools.invoke(messages)

        # Log if agent wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"   ✓ Agent decided to use {len(response.tool_calls)} tool(s)")
            for i, tool_call in enumerate(response.tool_calls, 1):
                logger.info(f"      {i}. {tool_call['name']}")
        else:
            logger.info("   ✓ Agent generated final response (no tools needed)")

        return {"messages": [response]}

    # Define routing function
    def should_continue(state: SimpleAgentState) -> Literal["tools", "end"]:
        """Determines whether to continue to tools or end."""
        last_message = state["messages"][-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.info("🔀 ROUTING: Continuing to TOOLS node")
            return "tools"


        logger.info("🔀 ROUTING: Ending workflow (final response ready)")
        return "end"

    # Create workflow
    workflow = StateGraph(SimpleAgentState)

    # Create tool node with logging
    original_tool_node = ToolNode(tools)

    def tool_node_with_logging(state):
        logger.info("🔧 TOOL NODE: Executing tools...")
        result = original_tool_node.invoke(state)
        logger.info(f"   ✓ Tools executed successfully")
        return result

    # Add nodes to graph
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_with_logging)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile graph
    if with_memory:
        logger.info("💾 Enabling conversation memory")
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)
    else:
        logger.info("⚠️  Memory disabled - stateless mode")
        graph = workflow.compile()

    logger.info("✅ Enhanced agent created successfully\n")
    return graph



# In[41]:

# In[46]:




config = {"configurable": {"thread_id": "ranking_test_1"}}
#result = enhanced_agent.invoke(
   # {"messages": [HumanMessage(content=custom_query)]},
    #config=config
#)

#print("\n🤖 AGENT RESPONSE:")
#print("="*80)
#print(result["messages"][-1].content)
#print("\n" + "="*80)

# =========================
# Production entry function
# =========================

_enhanced_agent = None

def get_enhanced_agent():
    """
    Lazy-init the agent so importing app.py never triggers heavy work.
    Cloud Run imports this module at startup; we must keep import side-effects minimal.
    """
    global _enhanced_agent
    if _enhanced_agent is None:
        _enhanced_agent = create_enhanced_financial_agent(with_rag=True, with_memory=True)
    return _enhanced_agent


def run_agent(payload: dict) -> dict:
    """
    Payload example:
      {"query": "Analyze NVDA and its AI initiatives"}

    Returns JSON-serializable dict.
    """
    query = (payload or {}).get("query", "").strip()
    if not query:
        return {"error": "Missing 'query' in request body."}

    agent = get_enhanced_agent()

    # Keep thread_id stable if you want memory per user/session; otherwise use a fixed value.
    thread_id = (payload or {}).get("thread_id", "cloudrun_default")
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    # Return the last assistant message content
    answer = result["messages"][-1].content if result.get("messages") else ""
    return {"answer": answer}


# =========================
# Optional local-only tests
# =========================
if __name__ == "__main__":
    # This section runs ONLY when you execute `python app.py` locally.
    # It will NOT run on Cloud Run import.
    demo = {"query": "Provide an investment analysis for NVDA, focusing on AI initiatives."}
    print(run_agent(demo))

#---17/02/2026---

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent()  # your existing function that builds the langgraph agent
    return _agent

def run_agent(query: str) -> str:
    agent = get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # adjust depending on your result format:
    return str(result)
