# BiztelAI Data Science Assignment

This project implements a data processing pipeline and REST API for analyzing chat transcripts from the BiztelAI dataset.

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sid0702/BiztelAI.git
   cd BiztelAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   > **Note:** Ensure `werkzeug==1.0.1` is installed for compatibility with `flask==2.0.1`.

3. **Prepare the dataset**:  
   Place the file `BiztelAI_DS_Dataset_V1.json` in the `data/` directory.

## Running the Application

Start the Flask API:
```bash
python app.py
```

### Access the API Endpoints

- `GET /` — Welcome message with available endpoints.
- `GET /summary` — Fetch dataset summaries and visualization paths.
- `POST /transform` — Preprocess a new transcript.
- `POST /analyze` — Analyze a transcript for insights.

## Example Usage

**Welcome Message:**
```bash
http://localhost:5000
```

**Get Summary:**
```bash
http://localhost:5000/summary
```

## Project Structure

- `app.py`: Flask application with API endpoints.
- `data/`: Contains the dataset.
- `scripts/`: Data processing and analysis scripts.
  - `data_loader.py`: Data ingestion and preprocessing.
  - `eda.py`: Exploratory data analysis and visualizations.
  - `sentiment.py`: Sentiment analysis using DistilBERT.
- `static/`: Stores visualization outputs (e.g., `word_cloud.png`, `message_length_distribution.png`).
- `templates/`: For potential HTML templates.

## Insights

The dataset contains chat transcripts discussing Washington Post articles. Agents discuss sports, politics, and media, with sentiments ranging from curious to happy.

**Visualizations include:**
- **Sentiment Distribution** — Shows sentiment breakdown by agent.
- **Message Count by Article** — Displays message volume per article.
- **Word Cloud** — Visualizes frequent words in messages.
- **Message Length Distribution** — Analyzes message length by agent.
- **Knowledge Source Frequency** — Shows common knowledge sources.
- **Sentiment Over Turn** — Tracks sentiment trends within transcripts.

> Sentiment analysis uses `distilbert-base-uncased-finetuned-sst-2-english` for accurate results.

