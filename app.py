#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required packages
# Run this cell first before executing the rest of the notebook

get_ipython().system('pip install    langchain==0.3.27    langchain-core==0.3.79    langchain-openai==0.3.11    langchain-community==0.3.31    langgraph==0.3.7    tavily-python    yfinance==0.2.66    chromadb==1.3.4    pypdf==6.2.0    tiktoken==0.12.0')

print("✅ All packages installed successfully!")


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

print("✅ All imports successful!")


# In[4]:


# Let Google Colab access my google drive
from google.colab import drive
drive.mount('/content/drive')


# In[5]:


import json
# Load the JSON file and extract values
file_name = "/content/drive/MyDrive/JHU_AI/Project_2/config.json"
with open(file_name, 'r') as file:
    config = json.load(file)


os.environ["OPENAI_API_KEY"] = config.get("API_KEY")
os.environ["OPENAI_API_BASE"] = config.get("OPENAI_API_BASE")
os.environ["TAVILY_API_KEY"] = config.get("TAVILY_API_KEY")

print("✅ Configuration loaded successfully!")
print(f"   Using API base: {os.environ['OPENAI_API_BASE']}")


# ---
# 
# # Section 1.1: The Goal (Proactiveness)
# 
# ## From Passive LLM to Proactive Agent
# 
# The key difference between a traditional LLM and an autonomous agent is **proactiveness**. Let's see this in action.

# In[19]:


# Traditional LLM Prompt (Reactive)
# This is a typical prompt for a traditional LLM - simple, reactive, and passive.

TRADITIONAL_PROMPT = """You are a helpful assistant.
Answer the user's question about stock information."""

print("📋 Traditional LLM Prompt:")
print("="*80)
print(TRADITIONAL_PROMPT)
print("="*80)
print("\n❌ Problems with this approach:")
print("   • No initiative - waits for user to specify what they want")
print("   • No comprehensive analysis - just answers the question")
print("   • Asks follow-up questions instead of taking action")
print("   • Provides minimal information")


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

print("📋 Agent Charter (Goal-Oriented):")
print("="*80)
print(AGENT_CHARTER_BASIC)
print("="*80)
print("\n✅ Benefits of this approach:")
print("   • Defines a clear mission and goal")
print("   • Specifies expected output format")
print("   • Encourages proactive information gathering")
print("   • Focuses on actionable insights, not just answers")


# In[6]:


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

print("✅ Tool 1: get_stock_price() - Defined")
print("   Purpose: Fetch real-time stock price and basic metrics")
print("   Data Source: Yahoo Finance (yfinance)")


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

print("✅ Tool 2: get_stock_history() - Defined")
print("   Purpose: Fetch historical performance for trend analysis")
print("   Key Metric: 3-year return percentage")


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

print("✅ Tool 3: search_financial_news() - Defined")
print("   Purpose: Search real-time financial news")
print("   Data Source: Tavily Search API")


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

print("✅ Tool 4: analyze_sentiment() - Defined")
print("   Purpose: Analyze sentiment of financial text")
print("   Method: OpenAI GPT-4 with keyword fallback")


# In[10]:


# Test get_stock_price manually
print("🧪 Testing get_stock_price tool with ticker 'AAPL':\n")
result = get_stock_price.invoke({"ticker": "AAPL"})
print(json.dumps(result, indent=2))

print("\n" + "="*80 + "\n")

# Test get_stock_history manually
print("🧪 Testing get_stock_history tool with ticker 'AAPL' and period '3y':\n")
result = get_stock_history.invoke({"ticker": "AAPL", "period": "3y"})
print(json.dumps(result, indent=2))


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

print("📋 Full Agent Charter with Constraints:")
print("="*80)
print("Key features:")
print("✅ Proactive Behavior: Takes initiative, gathers comprehensive data")
print("✅ Reactive Behavior: Handles tool failures, adapts to missing data")
print("✅ Autonomous Behavior: Makes independent judgments, cites sources")
print("✅ Quality Standards: Structured output with confidence levels")
print("="*80)


# In[12]:


# Building the Agent
# Agent State Definition
class SimpleAgentState(TypedDict):
    """
    State for the financial research agent.
    Tracks the conversation history with message accumulation.
    """
    messages: Annotated[Sequence, add_messages]

print("✅ Agent state defined")


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


# Testing the Agent
# Traditional LLM (Reactive)

print("="*80)
print("TEST 1: Traditional Reactive LLM")
print("="*80 + "\n")

# Create traditional agent
traditional_agent = create_financial_agent(agent_type="traditional", with_memory=False)

# Test query
query = "Tell me about Apple stock"
print(f"Query: {query}\n")
print("-"*80 + "\n")

# Run agent
result = traditional_agent.invoke({"messages": [HumanMessage(content=query)]})

print("\n🤖 TRADITIONAL LLM RESPONSE:")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)
print("\n❌ Notice: The traditional LLM may ask follow-up questions instead of taking action!")


# In[24]:


# Basic Autonomous Agent (Goal-Oriented)
print("="*80)
print("TEST 2: Basic Autonomous Agent (Goal-Oriented)")
print("="*80 + "\n")

# Create basic agent
basic_agent = create_financial_agent(agent_type="basic", with_memory=False)

# Test query
query = "Tell me about Apple stock"
print(f"Query: {query}\n")
print("-"*80 + "\n")

# Run agent
result = basic_agent.invoke({"messages": [HumanMessage(content=query)]})

print("\n🤖 BASIC AGENT RESPONSE:")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)
print("\n✅ Notice: The agent takes initiative and uses tools to gather data!")


# In[25]:


# Test 3: Full Autonomous Agent (With All Constraints)

print("="*80)
print("TEST 3: Full Autonomous Agent (With All Constraints)")
print("="*80 + "\n")

# Create full agent
full_agent = create_financial_agent(agent_type="full", with_memory=True)

# Test query
query = "Provide a comprehensive investment analysis for Microsoft (MSFT) including 3-year performance and AI research activity"
print(f"Query: {query}\n")
print("-"*80 + "\n")

# Run agent with memory
config = {"configurable": {"thread_id": "test_session_1"}}
result = full_agent.invoke(
    {"messages": [HumanMessage(content=query)]},
    config=config
)

print("\n🤖 FULL AGENT RESPONSE:")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)
print("\n✅ Notice: The agent provides comprehensive analysis with:")
print("   • Source citations")
print("   • Multiple tool usage")
print("   • Risk assessment")
print("   • Clear recommendation with confidence")
print("   • Data gap acknowledgment")


# In[26]:


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

print("⚠️  Created failing version of get_stock_price for testing")
print("   This will help us observe the agent's error handling behavior")


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

print("✅ Agent with failing tool creation function defined")


# In[28]:


# Test Error Handling

print("="*80)
print("TEST 4: Error Handling and Reactivity")
print("="*80 + "\n")
print("⚠️  Simulating tool failure: get_stock_price will return an error")
print("   Let's see how the agent reacts...\n")
print("-"*80 + "\n")

# Create agent with failing tool
failing_agent = create_agent_with_failing_tool()

# Test query
query = "Analyze Apple stock (AAPL)"
print(f"Query: {query}\n")
print("-"*80 + "\n")

# Run agent
result = failing_agent.invoke({"messages": [HumanMessage(content=query)]})

print("\n🤖 AGENT RESPONSE (with tool failure):")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)
print("\n✅ Observe how the agent:")
print("   • Detects the tool failure")
print("   • Tries alternative approaches (get_stock_history, news search)")
print("   • Acknowledges the data gap in the report")
print("   • Continues analysis with available information")
print("   • Adjusts confidence level based on incomplete data")


# In[29]:


# Test Your Own Queries

my_agent = create_financial_agent("full", with_memory=True)

# Configure memory
config = {"configurable": {"thread_id": "my_test_session"}}

# Test with your own query
# Try these examples or create your own:
# - "Analyze Tesla stock"
# - "Compare Microsoft and Google AI initiatives"
# - "What are the risks in investing in NVIDIA?"
# - "Tell me about Amazon's financial performance"

YOUR_QUERY =  """Provide a comprehensive investment analysis for Resolve AI (RZLV)"""

print("="*80)
print("YOUR CUSTOM QUERY TEST")
print("="*80 + "\n")
print(f"Query: {YOUR_QUERY}\n")
print("-"*80 + "\n")

result = my_agent.invoke(
    {"messages": [HumanMessage(content=YOUR_QUERY)]},
    config=config
)

print("\n🤖 AGENT RESPONSE:")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)


# In[30]:


# Additional imports for RAG implementation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

print("✅ RAG libraries imported successfully!")
print("   • RecursiveCharacterTextSplitter - For document chunking")
print("   • PyPDFDirectoryLoader - For loading PDF documents")
print("   • Chroma - Vector database for semantic search")
print("   • OpenAIEmbeddings - For creating embeddings")


# In[31]:


# Unzipping the AI Initiatives Documents
import zipfile
with zipfile.ZipFile("/content/drive/MyDrive/JHU_AI/Project_2/Companies-AI-Initiatives.zip", 'r') as zip_ref:
  zip_ref.extractall("/content/")         # Storing all the unzipped contents in this location


# In[32]:


# Path of all AI Initiative Documents
ai_initiative_pdf_paths = [f"/content/Companies-AI-Initiatives/{file}" for file in os.listdir("/content/Companies-AI-Initiatives")]
ai_initiative_pdf_paths


# In[33]:


from langchain_community.document_loaders import PyPDFDirectoryLoader
loader = PyPDFDirectoryLoader(path = "/content/Companies-AI-Initiatives/")          # Creating an PDF loader object


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


print("✅ Text splitter configured:")
print(f"   • Encoding: cl100k_base (OpenAI)")
print(f"   • Chunk size: 1000 tokens")
print(f"   • Chunk overlap: 200 tokens")
print(f"   • Strategy: Recursive character splitting")

# Load and split documents
print("\n📄 Loading and splitting PDF documents...")
ai_initiative_chunks = loader.load_and_split(text_splitter)

print(f"✅ Documents processed successfully!")
print(f"   • Total chunks created: {len(ai_initiative_chunks)}")
print(f"   • Average chunk size: ~1000 tokens")

# Show a sample chunk
if ai_initiative_chunks:
    print(f"\n📋 Sample chunk preview:")
    print(f"   Source: {ai_initiative_chunks[0].metadata.get('source', 'Unknown')}")
    print(f"   Content preview: {ai_initiative_chunks[0].page_content[:200]}...")


# In[35]:


# Step 4: Create Vector Store with Embeddings - create embeddings for each chunk and store them in ChromaDB for semantic search.

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# Initialize OpenAI embedding model (text-embedding-ada-002)
embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

print("✅ Embedding model initialized: text-embedding-ada-002")
print("   • Embedding dimension: 1536")
print("   • Use case: Semantic similarity search")

# Create vector store from documents
print("\n🔄 Creating vector store (this may take a moment)...")
print("   Generating embeddings for all chunks...")

vectorstore = Chroma.from_documents(
    ai_initiative_chunks,
    embedding_model,
    collection_name="AI_Initiatives"
)

print(f"\n✅ Vector store created successfully!")
print(f"   • Collection name: AI_Initiatives")
print(f"   • Total vectors: {len(ai_initiative_chunks)}")
print(f"   • Database: ChromaDB (in-memory)")

# Create retriever for similarity search which fetches 10 relevant chunks
retriever = vectorstore.as_retriever(
    search_type= 'similarity',
    search_kwargs={'k': 10}
)

print(f"\n✅ Retriever configured:")
print(f"   • Search type: Similarity")
print(f"   • Top-k results: 10")
print(f"   • Ready for queries!")


# In[36]:


# Test query - Let's test the retrieval system manually before integrating it with the agent.
test_query = "What AI projects is Microsoft working on?"

print(f"🔍 Test Query: {test_query}")
print("="*80)

# Retrieve relevant documents
relevant_docs = retriever.get_relevant_documents(test_query)

print(f"\n✅ Retrieved {len(relevant_docs)} relevant chunks:\n")

# Display top 3 results
for i, doc in enumerate(relevant_docs[:3], 1):
    print(f"📄 Result {i}:")
    print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"   Page: {doc.metadata.get('page', 'Unknown')}")
    print(f"   Content: {doc.page_content[:300]}...")
    print("-"*80 + "\n")


# In[37]:


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

        # Build the full prompt
        formatted_prompt = f"""[INST]{qna_system_message}

                {'user'}: {qna_user_message_template.format(context=context_for_query, question=query)}
                [/INST]"""

        # Query the LLM
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        response = model.invoke(formatted_prompt)

        return response.content

    except Exception as e:
        return f"Error querying private database: {str(e)}"

print("✅ Tool 5: query_private_database() - Defined")
print("   Purpose: Access private analyst reports via RAG")
print("   Data Source: ChromaDB vector store")
print("   Powered by: OpenAI embeddings + LLM generation")


# In[38]:


# Test the query_private_database tool
test_queries = [
    "What AI projects is Microsoft working on?",
    "What are NVIDIA's AI research areas?",
    "Tell me about Google's AI initiatives"
]

for test_query in test_queries:
    print("="*80)
    print(f"🔍 Query: {test_query}\n")
    result = query_private_database.invoke({"query": test_query})
    print(f"📄 Answer:\n{result}")
    print("\n")


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

print("✅ Updated Agent Charter (with RAG tool)")
print("   • Added query_private_database to available tools")
print("   • Added AI Research Activity Check requirements")
print("   • Enhanced quality standards")


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

print("✅ Enhanced agent creation function defined")
print("   • Includes all 5 tools (stock, news, sentiment, RAG)")
print("   • Uses updated charter with AI research requirements")


# In[41]:


# Test 5: Agent with RAG - AI Research Activity Check

# Let's test the enhanced agent's ability to analyze companies with AI research insights.

print("="*80)
print("TEST 5: Enhanced Agent with RAG - AI Research Activity Check")
print("="*80 + "\n")

# Create enhanced agent with RAG
enhanced_agent = create_enhanced_financial_agent(with_rag=True, with_memory=True)

# Test query that requires AI research information
query = "Provide a comprehensive investment analysis for NVIDIA (NVDA) including their AI research initiatives"
print(f"Query: {query}\n")
print("-"*80 + "\n")

# Run agent with memory
config = {"configurable": {"thread_id": "enhanced_test_1"}}
result = enhanced_agent.invoke(

    {"messages": [HumanMessage(content=query)]},
    config=config
)

print("\n🤖 ENHANCED AGENT RESPONSE (with RAG):")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)
print("\n✅ Notice: The agent now includes:")
print("   • AI research projects from private analyst reports")
print("   • Specific AI initiative details")
print("   • Integration of financial + AI research data")
print("   • Comprehensive investment recommendation")


# In[43]:


# Demonstrate how the agent uses multiple tools together: news search → sentiment analysis → RAG query.
print("="*80)
print("TEST 6: Synergistic Tool Usage (News + Sentiment + RAG)")
print("="*80 + "\n")

query = "Analyze Microsoft's position in the AI market. Include recent news sentiment and their strategic AI initiatives."
print(f"Query: {query}\n")
print("-"*80 + "\n")

config = {"configurable": {"thread_id": "synergy_test_1"}}
result = enhanced_agent.invoke(
    {"messages": [HumanMessage(content=query)]},
    config=config
)

print("\n🤖 AGENT RESPONSE (Synergistic Tool Usage):")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)
print("\n✅ The agent demonstrated synergistic tool usage:")
print("   1. search_financial_news() - Found recent articles")
print("   2. analyze_sentiment() - Analyzed news sentiment")
print("   3. query_private_database() - Retrieved AI initiative details")
print("   4. get_stock_history() - Got financial performance")
print("   5. Synthesized all data into comprehensive report")


# In[45]:


# Rank companies based on both financial performance AND AI research activity.
print("="*80)
print("TEST 7: Investment Recommendation System - Multi-Company Ranking")
print("="*80 + "\n")

# Define companies to analyze
companies = ["MSFT", "GOOGL", "NVDA", "AMZN", "IBM"]

query = f"""

You are an autonomous financial analyst. Perform a comparative investment
analysis across the following companies. For each company, you MUST:
1. Retrieve stock performance (current + 3-year history)
2. Analyze recent financial news and sentiment
3. Query private database for AI research initiatives
4. Identify risks and opportunities
5. Produce a Buy/Hold/Sell recommendation with confidence
6. Provide proper citations for all sources

Companies to analyze: {', '.join(companies)}


After analyzing each company individually, produce:
1. A ranked list from strongest to weakest investment opportunity
2. A short justification for each ranking position
3. A final summary: “Best AI-sector investment among these companies is ___ because…”
4. NOTE: Use consistent evaluation criteria for fairness (financial strength, AI activity, sentiment, risk)

"""

print(f"Query: Multi-company investment ranking\n")
print(f"Companies: {', '.join(companies)}\n")
print("-"*80 + "\n")

config = {"configurable": {"thread_id": "ranking_test_1"}}
result = enhanced_agent.invoke(
    {"messages": [HumanMessage(content=query)]},
    config=config
)

print("\n🤖 INVESTMENT RANKING REPORT:")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)


# In[46]:


# Final Interactive Test Cell
# Try your own custom queries here!

#custom_query = "Which public company currently leads the industry in innovative AI research? Provide evidence from private analyst reports, recent financial news, and any available AI initiative documents. Include citations, sentiment analysis, and a recommendation backed by confidence scoring."
#custom_query = "Compare NVIDIA and AMD in the context of AI infrastructure leadership. Include stock performance, news sentiment, AI research projects (from private database), competitive advantages, risks, and a final recommendation with confidence levels."
custom_query = "Find a public company with strong financial performance but underreported AI initiatives. Use stock history, news sentiment, and private AI reports. Explain why their AI innovation is underreported, and whether this represents an investment opportunity."
print("="*80)
print("YOUR CUSTOM QUERY")
print("="*80 + "\n")
print(f"Query: {custom_query}\n")
print("-"*80 + "\n")

config = {"configurable": {"thread_id": "ranking_test_1"}}
result = enhanced_agent.invoke(
    {"messages": [HumanMessage(content=custom_query)]},
    config=config
)

print("\n🤖 AGENT RESPONSE:")
print("="*80)
print(result["messages"][-1].content)
print("\n" + "="*80)

