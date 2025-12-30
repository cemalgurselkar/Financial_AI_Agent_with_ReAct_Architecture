import yfinance as yf
from duckduckgo_search import DDGS
import pandas as pd
import requests
import datetime
import json
import os
from knowledge_base import KnowledgeBase
from translate import TranslatorWrapper

kb = KnowledgeBase()
translator = TranslatorWrapper()
ddgs = DDGS()

# import pandas as pd ve import os olduğundan emin ol.

def analyze_full_csv(file_path: str) -> str:
    """
    Reads a CSV file and performs detailed STATISTICAL and TREND analysis for the LLM.
    Processes numeric data to extract growth rates and risk (volatility) metrics.
    """
    try:
        file_path = file_path.strip().strip('"').strip("'")
        if not os.path.exists(file_path):
            return f"ERROR: '{file_path}' not found."

        df = pd.read_csv(file_path)
        if df.empty: return "File is empty."

        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        if numeric_df.empty:
            return "File read, but no numeric data (price, quantity, etc.) found to analyze."

        stats = numeric_df.describe().to_string()

        trend_report = []
        
        for col in numeric_df.columns:
            series = numeric_df[col]
            start_val = series.iloc[0]
            end_val = series.iloc[-1]

            if start_val != 0:
                change_pct = ((end_val - start_val) / start_val) * 100
            else:
                change_pct = 0.0
            
            trend_direction = "UPTREND" if change_pct > 0 else "DOWNTREND"

            mean_val = series.mean()
            volatility = (series.std() / mean_val) * 100 if mean_val != 0 else 0
            
            trend_report.append(
                f"--- COLUMN: {col} ---\n"
                f"   • Start Value: {start_val}\n"
                f"   • End Value: {end_val}\n"
                f"   • Total Change: {change_pct:.2f}% ({trend_direction})\n"
                f"   • Volatility (Risk): {volatility:.2f}% (Higher value = Higher fluctuation)\n"
            )

        trend_text = "\n".join(trend_report)

        correlation_text = ""
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr().to_string()
            correlation_text = f"\n\n--- CORRELATION MATRIX ---\n{corr_matrix}\n(1.00 = Perfect correlation, -1.00 = Inverse correlation, 0 = No correlation)"

        final_output = (
            f"DETAILED DATA ANALYSIS REPORT:\n"
            f"Total Rows: {len(df)}\n\n"
            f"1. BASIC STATISTICS:\n{stats}\n\n"
            f"2. TREND AND RISK ANALYSIS:\n{trend_text}"
            f"{correlation_text}"
        )
        
        return final_output

    except Exception as e:
        return f"Analysis Error: {str(e)}"


def read_csv_preview(file_path: str) -> str:
    """
    Yerel bir CSV dosyasını okur ve içeriğini metin olarak döndürür.
    Analiz yapmaz, sadece veriyi sunar.
    """
    try:
        file_path = file_path.strip().strip('"').strip("'")
        if not os.path.exists(file_path):
            return f"HATA: '{file_path}' dosyası bulunamadı. Lütfen dosya adını kontrol et."

        df = pd.read_csv(file_path)
        
        if df.empty:
            return "Dosya boş."

        info = f"DOSYA BİLGİSİ:\n- Kolonlar: {list(df.columns)}\n- Toplam Satır: {len(df)}\n\n"

        head_data = df.head(5).to_string(index=False)
        tail_data = df.tail(5).to_string(index=False)
        
        return f"{info}--- İLK 5 SATIR ---\n{head_data}\n\n--- SON 5 SATIR ---\n{tail_data}"

    except Exception as e:
        return f"CSV Okuma Hatası: {str(e)}"

SERPER_API_KEY = "10ecd4cb2c80de28130888e1b6eb6581b8440632"

def search_general_info(query: str) -> str:
    """
    Google üzerinden en güncel ve yerel finansal haberleri arar (Serper API).
    """
    try:
        url = "https://google.serper.dev/search"
        
        # Türkiye odaklı arama yapıyoruz (gl=tr, hl=tr)
        # Bu sayede "THY Başkanı" gibi sorulara Türkçe ve doğru cevap gelir.
        payload = json.dumps({
            "q": query,
            "gl": "tr",
            "hl": "tr",
            "num": 4  # Kaç sonuç gelsin?
        })
        
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json()

        summary = ""

        # 1. "Answer Box" (Direkt Cevap Kutusu) varsa onu en başa koy
        # Örn: "TCMB Faizi" yazınca Google'ın en üstte çıkardığı %50 kutusu.
        if "answerBox" in results:
            box = results["answerBox"]
            answer = box.get("answer") or box.get("snippet") or box.get("title")
            if answer:
                summary += f"✅ [DİREKT CEVAP]: {answer}\n\n"

        # 2. Organik Sonuçlar
        if "organic" in results:
            for res in results["organic"]:
                title = res.get('title', '')
                snippet = res.get('snippet', '')
                date = res.get('date', '') # Varsa tarih
                summary += f"- {title} ({date}): {snippet}\n"
        
        if not summary:
            return "Google'da ilgili sonuç bulunamadı."

        return summary

    except Exception as e:
        return f"Google Arama Hatası: {str(e)}"

def get_ticker_symbol(company_name: str) -> str:
    """
    Şirket isminden sembolü Yahoo Finance API'si üzerinden dinamik olarak bulur.
    Arama motoru kullanmaz, bu yüzden çok hızlıdır ve patlamaz.
    """
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params = {
            'q': company_name,
            'quotesCount': 1,
            'newsCount': 0,
            'enableFuzzyQuery': 'false',
            'quotesQueryId': 'tss_match_phrase_query'
        }

        response = requests.get(url, headers=headers, params=params, timeout=5)
        
        if response.status_code != 200:
            return "Yahoo sunucusuna ulaşılamadı."

        data = response.json()

        if 'quotes' in data and len(data['quotes']) > 0:
            best_match = data['quotes'][0]
            symbol = best_match.get('symbol')
            shortname = best_match.get('shortname', symbol)
            exch = best_match.get('exchange', 'Unknown')

            if exch == 'IST' and not symbol.endswith('.IS'):
                symbol += ".IS"

            return f"Bulunan Sembol: {symbol} (Şirket: {shortname}, Borsa: {exch})"
        
        return "Şirket/Coin bulunamadı. Lütfen ismin doğruluğunu kontrol et."

    except Exception as e:
        return f"Sembol bulma hatası: {str(e)}"

def get_stock_price(ticker: str) -> str:
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        
        df = stock.history(period="5d")
        
        if df.empty:
            return f"HATA: '{ticker}' için veri yok. Sembol hatalı olabilir."
        
        last_row = df.iloc[-1]
        last_price = last_row['Close']
        last_date = df.index[-1].strftime("%Y-%m-%d")
        
        return f"DATA:\nSembol: {ticker}\nTarih: {last_date}\nFiyat: {last_price:.2f}"

    except Exception as e:
        return f"Fiyat hatası: {str(e)}"

def analyze_technical_data(ticker: str) -> str:
    try:
        ticker = ticker.strip().upper()
        df = yf.download(ticker, period="1y", progress=False)
        
        if df.empty or len(df) < 50:
            return "Teknik analiz için yeterli veri yok."

        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        sma_50 = close.rolling(window=50).mean().iloc[-1]
        sma_200 = close.rolling(window=200).mean().iloc[-1]
        
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]

        current_price = close.iloc[-1]
        trend = "YÜKSELİŞ" if current_price > sma_200 else "DÜŞÜŞ"

        return (
            f"TEKNİK ANALİZ ({ticker}):\n"
            f"Fiyat: {current_price:.2f}\n"
            f"RSI (14): {last_rsi:.2f}\n"
            f"SMA 50: {sma_50:.2f}\n"
            f"Trend: {trend}"
        )

    except Exception as e:
        return f"Analiz hatası: {str(e)}"

# --- 5. RAG ---
def query_knowledge_base(query: str) -> str:
    try:
        query_en = translator.translate_to_en(query)
        results = kb.retrieve(query_en, mode="auto")
        return f"BİLGİ BANKASI:\n{results}"
    except Exception as e:
        return f"DB Hatası: {str(e)}"
