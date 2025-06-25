# 📈 yFinance Stock Analyzer: Scraper, Visualizer & News Sentiment

A **Python-based stock analysis suite** that:
- Connects to **Yahoo Finance** to fetch historical stock data.
- Syncs **NASDAQ-100 and TSX-60 tickers** into a local SQLite3 database.
- Fetches and analyzes **recent news headlines** for each stock, using advanced sentiment analysis (TextBlob and transformer models).
- Stores **industry and sector info** for each stock.
- Provides **buy/sell recommendations** based on technical indicators, volume, news sentiment, and industry-relative momentum.
- Plots detailed charts for each recommended stock, including price, moving averages, RSI, volume, and sentiment annotation.
- Supports **multi-threaded scraping** for fast data sync.
- Can be run and deployed via **Docker**.

---

## 🚀 Features

- ✅ Batch downloading with multi-threading for fast syncing.
- ✅ Supports both **NASDAQ-100** and **TSX-60** tickers.
- ✅ Incremental sync: only new data is pulled.
- ✅ News headlines and sentiment scores (TextBlob or transformer) are stored for each stock.
- ✅ Industry and sector info is fetched and stored.
- ✅ Buy/sell scoring system based on price, moving averages, RSI, momentum, volume, news sentiment, and industry comparison.
- ✅ Generates detailed matplotlib and Plotly charts for recommended stocks, with sentiment overlay.
- ✅ All stock lines can be plotted together for comparison.
- ✅ Modular architecture: scraping, analysis, news, and sentiment are in separate modules.
- ✅ Docker support for easy deployment and reproducibility.

---

## 📂 Project Structure

```text
yFinance-stock-analyzer/
│
├── app/
│   ├── stockScraper.py         # Main data sync script (prices, news, industry)
│   ├── stockAnalyzer.py        # Analysis, scoring, and plotting
│   ├── stockSimulator.py       # Portfolio simulation and backtesting
│   ├── webui.py                # Flask web UI for visualization
│   ├── database_manager.py     # Centralized DB manager
│   ├── appconfig.py            # Configuration constants
│   └── news_analyzer/
│       ├── __init__.py
│       ├── data_fetcher.py
│       ├── industry_analyzer.py
│       ├── news_processor.py
│       └── sentiment_analyzer.py
│
├── tests/                      # Unit tests for all modules
│   └── news_analyzer/
│       ├── test_data_fetcher.py
│       ├── test_industry_analyzer.py
│       ├── test_news_processor.py
│       └── test_sentiment_analyzer.py
│   ├── test_stockAnalyzer.py
│   ├── test_databaseManager.py
│   └── test_stockSimulator.py
│
├── data/                       # (Optional) Local data directory, can be mounted in Docker
├── plots/                      # Output charts
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker build instructions
├── README.md                   # Project documentation
└── stocks.db                   # SQLite database (auto-created)
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

---

## ⚙️ How to Run

### 1. Sync Data (prices, news, industry)

```bash
python app/stockScraper.py
```
- Downloads historical prices, news, and industry info for all tickers.
- Populates `stocks.db`.

### 2. Analyze and Visualize

```bash
python app/stockAnalyzer.py
```
- Scores each stock for buy/sell based on technicals, sentiment, and industry.
- Prints top buy/sell recommendations with sentiment and industry info.
- Saves detailed charts for each recommended stock in a `plots/` folder.

### 3. Run the Web UI

```bash
python app/webui.py
```
- Starts a Flask server for interactive visualization at [http://localhost:5000](http://localhost:5000).

---

## 🐳 Docker Usage

### 1. Build the Docker Image

```bash
docker build -t yfinance-analyzer .
```

### 2. Run the Container (with data directory mounted and port exposed)

```bash
docker run --rm -it -v "$PWD/data":/data -p 5000:5000 yfinance-analyzer
```
- On Windows CMD: `-v %cd%\data:/data`
- On PowerShell: `-v ${PWD}\data:/data`

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
- News sentiment uses [TextBlob](https://textblob.readthedocs.io/en/dev/) and can optionally use transformer models (FinBERT, RoBERTa).
- For best Docker performance, mount your data directory from within your WSL/Linux home directory (not `/mnt/c/...`).

---

## ✅ Dependencies

- Python 3.8 or higher
- yfinance
- pandas
- matplotlib
- tqdm
- textblob
- flask
- transformers
- torch
- plotly
- lxml

---

## ✨ Optional Improvements

- Add filtering to sync only a subset of tickers.
- Use more advanced sentiment models (e.g., transformers).
- Add sector-based coloring to plots.
- Convert charts to interactive Plotly graphs.
- Schedule regular syncs with a task scheduler.
- Improve buy/sell scoring algorithms with more metrics (volatility, Sharpe ratio, etc.).
- Add robust error handling and retry logic for yfinance scraping.

---

## 🧪 Testing

- Unit tests are provided for all major modules.
- Run tests with:
  ```bash
  pytest
  ```

---

Feel free to ask for help with Dockerization, packaging, or extending the analysis!