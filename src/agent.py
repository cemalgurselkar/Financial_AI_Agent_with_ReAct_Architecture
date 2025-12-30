import ollama
import re
import tools 

# Colorama kütüphanesini sildik. Artık [0m gibi bozuk karakterler çıkmayacak.

class FinancialAgent:
    def __init__(self):
        self.model = "qwen2.5:3b-instruct" 
        print(f"🤖 Financial ReAct Agent Initialized (Model: {self.model})")

        self.max_turns = 7

        self.tools_map = {
            "search_general_info": tools.search_general_info,
            "get_ticker_symbol": tools.get_ticker_symbol,
            "get_stock_price": tools.get_stock_price,
            "analyze_technical_data": tools.analyze_technical_data,
            "query_knowledge_base": tools.query_knowledge_base,
            "read_csv_preview": tools.read_csv_preview,
            "analyze_full_csv": tools.analyze_full_csv
        }

        self.system_prompt = """
You are a Senior Financial Research Agent.

AVAILABLE TOOLS (USE ONLY THESE):
1. search_general_info: Searches the internet for current news, macroeconomic events, or specific details. Input: A search query string (e.g., 'Fed interest rate decision').
2. get_ticker_symbol: Identifies the stock ticker symbol for a public company. ALWAYS use this FIRST for company queries. Input: Company name (e.g., 'Aselsan').
3. get_stock_price: Retrieves the current live price and currency of a specific ticker. Input: Ticker symbol (e.g., 'ASELS.IS').
4. analyze_technical_data: Calculates technical indicators (RSI, SMA, Trend) for a ticker to assess market strength. Input: Ticker symbol.
5. query_knowledge_base: Searches the internal financial library (books) for definitions, investment strategies, and theories. NOT for live news. Input: Query string.
6. read_csv_preview: Reads the first 5 rows of a CSV file to understand its structure/columns. Input: File path.
7. analyze_full_csv: Performs deep statistical analysis (Trend, Volatility, Mean) on a CSV file. Input: File path.

CRITICAL RULES:
1. **NO FAKE TOOLS:** Do NOT invent tools like `calculate`, `summarize`, `convert_currency`.
2. **MATH:** Perform simple calculations (multiplication, division) MENTALLY in your 'Thought' process. Do not call a tool for math.
3. **EXACT TICKER:** If `get_ticker_symbol` returns 'ASELS.IS', use exactly 'ASELS.IS'.
4. **FORMAT:** Strictly use the format below. No parentheses in Action line.

Format Example:
Question: What is Aselsan price in USD?
Thought: I need the ticker first.
Action: get_ticker_symbol
Action Input: Aselsan
Observation: ASELS.IS
Thought: Now I get the price.
Action: get_stock_price
Action Input: ASELS.IS
Observation: 224.10 TL
Thought: I need USD/TRY rate to convert.
Action: get_stock_price
Action Input: TRY=X
Observation: 35.50
Thought: 224.10 / 35.50 is approx 6.31 USD. I have enough info.
Final Answer: Aselsan is trading at 224.10 TL, which is approximately 6.31 USD.

STRATEGY:
- For Company Analysis: Ticker -> Price -> Technicals -> News.
- For CSV Analysis: analyze_full_csv.
""".strip()

    def log(self, type, msg):
        """Artık renk kodları yok, sadece temiz metin."""
        print(f"[{type}]: {msg}")

    def _parse_action(self, text):
        """Güçlü Parser: Hatalı formatları düzeltir."""
        lines = text.split('\n')
        tool = None
        arg = None

        for line in lines:
            line = line.strip()
            if line.startswith("Action:") and not tool:
                raw_tool = line.split("Action:")[1].strip()
                # Temizlik
                if "(" in raw_tool: raw_tool = raw_tool.split("(")[0].strip()
                if "->" in raw_tool: raw_tool = raw_tool.split("->")[0].strip()
                tool = raw_tool

            elif line.startswith("Action Input:") and not arg:
                arg = line.split("Action Input:")[1].strip().strip('"').strip("'")

        # Yedek Regex (Model input satırını unuttuysa)
        if tool and not arg:
            match = re.search(r"Action:.*?\((.*?)\)", text)
            if match: arg = match.group(1).strip('"').strip("'")

        return tool, arg

    def generate_analyst_report(self, query, history_context):
        self.log("Thought", "Veriler toplandı, Analist raporu hazırlanıyor...")
        
        system_rules = """
        GÖREV: Profesyonel Finansal Analist olarak Türkçe rapor yaz.
        
        KURALLAR:
        1. SADECE TÜRKÇE konuş.
        2. Sayısal verileri (Fiyat, RSI) raporda mutlaka kullan.
        3. "Yatırım Tavsiyesi Değildir" (YTD) uyarısını ekle.
        4. Veri yoksa "Veriye ulaşılamadı" de, sayı uydurma.
        """
        
        user_input = f"SORU: {query}\n\nTOPLANAN VERİLER:\n{history_context}"
        
        response = ollama.chat(
            model=self.model, 
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": user_input}
            ],
            options={"temperature": 0.1, "num_ctx": 4096}
        )
        return response['message']['content']

    def run(self, user_query):
        self.log("User", user_query)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {user_query}"}
        ]
        
        full_history_log = "" 

        for step in range(self.max_turns):
            response = ollama.chat(
                model=self.model, 
                messages=messages,
                options={
                    "temperature": 0.0,
                    "num_ctx": 4096,
                    "stop": ["Observation:"]
                }
            )
            output = response['message']['content'].strip()

            if "Action:" in output:
                thought = output.split("Action:")[0].strip()
                self.log("Thought", thought)
            else:
                self.log("Thought", output)

            if "Final Answer:" in output:
                final_info = output.split("Final Answer:")[1].strip()
                full_history_log += f"\n[Sonuç]: {final_info}"
                report = self.generate_analyst_report(user_query, full_history_log)
                self.log("Final Answer", report)
                return report

            tool_name, tool_arg = self._parse_action(output)
            
            if tool_name:
                self.log("Action", f"{tool_name} -> {tool_arg}")
                
                if tool_name in self.tools_map:
                    try:
                        if not tool_arg:
                             tool_result = "Error: Missing argument."
                        else:
                            func = self.tools_map[tool_name]
                            tool_result = str(func(tool_arg))
                    except Exception as e:
                        tool_result = f"Tool Error: {str(e)}"
                else:
                    tool_result = f"Error: Tool '{tool_name}' not found."

                self.log("Observation", tool_result)

                full_history_log += f"\n[Adım {step+1}]\nİşlem: {tool_name}('{tool_arg}')\nVeri: {tool_result}\n"

                step_context = f"{output}\nObservation: {tool_result}"
                messages.append({"role": "assistant", "content": step_context})

            else:
                if "Final Answer" not in output:
                     messages.append({"role": "user", "content": "Please continue with an Action."})

        return "İşlem süresi doldu."

if __name__ == "__main__":
    agent = FinancialAgent()
    while True:
        try:
            q = input("\nSoru (q çıkış): ")
            if q.lower() == 'q': break
            print("\n" + "-"*30)
            agent.run(q)
            print("-"*30)
        except KeyboardInterrupt:
            break