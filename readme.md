# Financial AI Agent with ReAct Architecture

## Project Description
This project implements an autonomous "Financial Analyst" agent designed to overcome the static data limitations of traditional Large Language Models (LLMs). Built upon the ReAct (Reasoning + Acting) architecture, this system goes beyond simple text generation; it autonomously gathers real-time financial data, performs technical analysis, and provides logical reasoning for complex financial queries.

The system utilizes the Qwen 2.5 3B model (running locally via Ollama) as its cognitive engine, integrated with Google Serper API and YFinance libraries as external tools.

## Key Features

### 1. Real-Time Market Tracking
The agent fetches live price data from Borsa Istanbul (BIST), NASDAQ, and Cryptocurrency markets. Unlike standard models, it responds based on real-time market conditions rather than outdated training data.

### 2. Technical Analysis Capabilities
For stocks and crypto assets, the agent calculates key technical indicators such as RSI (Relative Strength Index) and SMA (Simple Moving Average) to interpret market signals (e.g., Overbought/Oversold).

### 3. Macroeconomic Data and News Retrieval
Through Google Serper API integration, the agent accesses live exchange rates, central bank interest rates, breaking economic news, and official company disclosures.

### 4. CSV Data Analysis
The system can process user-uploaded financial datasets (CSV format). It performs statistical analysis, including trend identification, volatility calculation, and correlation analysis between variables.

### 5. ReAct Loop Implementation
The model answers user queries by generating a Chain of Thought (CoT) rather than a direct response:
* Thought: What specific data is required to answer this?
* Action: Execute the relevant Python function/tool.
* Observation: Analyze the output from the tool.
* Final Answer: Synthesize the findings into a coherent response.

## System Architecture

The project consists of the following core components:

* **LLM Engine:** Qwen 2.5 3B Instruct (hosted locally via Ollama).
* **Orchestrator:** Python-based ReAct loop managing the flow between the LLM and tools.
* **Tools:**
    * `get_stock_price`: Fetches current asset prices.
    * `search_general_info`: Google Search wrapper for news and macro data.
    * `analyze_technical_data`: Technical indicator calculator.
    * `analyze_csv`: Statistical processor for local files.
* **User Interface:** Streamlit-based web interface.

## Performance and Benchmark Results

The table below compares the performance of our Local ReAct Agent against other leading models on the same financial dataset:

| Model / Architecture | Accuracy | Success Rate | Notes |
| :--- | :--- | :--- | :--- |
| **Financial AI Agent (This Project)** | **84%** | **42/50** | Uses Qwen 2.5 (Local) + Tools |
| Google Gemini (Advanced) | 96% | 48/50 | High reasoning capability |
| Groq (Llama 3 70B) | 94% | 47/50 | Extremely fast inference |
| Meta Llama 3 (Base) | 90% | 45/50 | Good performance but slower |

**Note:** Detailed logs and execution traces for these benchmark results are available in the `benchmark_logs/` directory within this repository.

## Installation and Usage

Follow these steps to run the project in a local environment.

### 1. Prerequisites
* Python 3.10 or higher
* Ollama (with Qwen 2.5 model pulled)
* Google Serper API Key

### 2. Install Dependencies
Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

## How to run
```python
streamlit run app.py
```