# 📈 Stock Analyzer: Scraper, Visualizer & News Sentiment

This project is a **Python-based stock analysis suite** that:
- Connects to **Yahoo Finance** to fetch historical stock data.
- Syncs **NASDAQ-100 and TSX-60 tickers** into a local SQLite3 database.
- Fetches and analyzes **recent news headlines** for each stock, using sentiment analysis.
- Stores **industry and sector info** for each stock.
- Provides **buy/sell recommendations** based on technical indicators, volume, news sentiment, and industry-relative momentum.
- Plots detailed charts for each recommended stock, including price, moving averages, RSI, volume, and sentiment annotation.

---

## 🚀 Features
- ✅ Batch downloading with multi-threading for fast syncing.
- ✅ Supports both **NASDAQ-100** and **TSX-60** tickers.
- ✅ Incremental sync: only new data is pulled.
- ✅ News headlines and sentiment scores are stored for each stock.
- ✅ Industry and sector info is fetched and stored.
- ✅ Buy/sell scoring system based on price, moving averages, RSI, momentum, volume, news sentiment, and industry comparison.
- ✅ Generates detailed matplotlib charts for recommended stocks, with sentiment overlay.
- ✅ All stock lines can be plotted together for comparison.

---

## 📂 Project Structure
```text
yFinance-stock-analyzer/
│
├── stockScraper.py     # Main data sync script (prices, news, industry)
├── stockAnalyzer.py    # Analysis, scoring, and plotting
├── newsAnalyzer.py     # News fetching, sentiment, and industry helpers
├── stocks.db           # SQLite database (auto-created)
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/yFinance-stock-analyzer.git
cd yFinance-stock-analyzer
```

### 2. Set Up Python Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate   # On Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
If you don’t have a `requirements.txt`, install manually:
```bash
pip install yfinance pandas matplotlib tqdm textblob
```

---

## ⚙️ How to Run

### 1. Sync Data (prices, news, industry)
```bash
python stockScraper.py
```
- Downloads historical prices, news, and industry info for all tickers.
- Populates `stocks.db`.

### 2. Analyze and Visualize
```bash
python stockAnalyzer.py
```
- Scores each stock for buy/sell based on technicals, sentiment, and industry.
- Prints top buy/sell recommendations with sentiment and industry info.
- Saves detailed charts for each recommended stock in a `plots/` folder.

---

## 🖼️ Example Outputs

- **Buy/Sell Recommendations:**  
  ```
  Buy recommendations:
  AAPL (Score: 6, Avg Sentiment: 0.23, Industry: Consumer Electronics)
  ...
  Sell recommendations:
  TSLA (Score: 5, Avg Sentiment: -0.15, Industry: Auto Manufacturers)
  ```
- **Charts:**  
  - Price, 20/50-day moving averages, RSI, volume, and sentiment annotation for each recommended stock.

---

## 📌 Notes
- **No manual input required.**
- Internet access is required to fetch data and news.
- The initial sync may take several minutes.
- The script is safe to run repeatedly — it will **incrementally update** the database.
- News sentiment uses [TextBlob](https://textblob.readthedocs.io/en/dev/) for simple polarity scoring.

---

## ✅ Dependencies
- Python 3.8 or higher
- yfinance
- pandas
- matplotlib
- tqdm
- textblob


---

## ✨ Optional Improvements
- Add filtering to sync only a subset of tickers.
- Use more advanced sentiment models (e.g., transformers).
- Add sector-based coloring to plots.
- Convert charts to interactive Plotly graphs.
- Schedule regular syncs with a task scheduler.

---

Feel free to ask for help with Dockerization, packaging,