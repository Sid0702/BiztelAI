BiztelAI Data Science Assignment
This project implements a data processing pipeline and REST API for analyzing chat transcripts from the BiztelAI dataset.
Setup

Clone the repository:
git clone <repository_url>
cd biztelai_assignment


Install dependencies:
pip install -r requirements.txt

Note: Ensure werkzeug==1.0.1 is installed for compatibility with flask==2.0.1.

Ensure the dataset BiztelAI_DS_Dataset_V1.json is in the data/ directory.


Running the Application

Start the Flask API:
python app.py


Access the API endpoints:

GET /: Welcome message with available endpoints.
GET /summary: Fetch dataset summaries and visualization paths.
POST /transform: Preprocess a new transcript.
POST /analyze: Analyze a transcript for insights.



Example Usage

Welcome Message:
curl http://localhost:5000


Get Summary:
curl http://localhost:5000/summary


Transform Data:
curl -X POST -H "Content-Type: application/json" -d '{"transcript_id": "test", "article_url": "http://example.com", "config": "A", "content": [{"message": "Test message", "agent": "agent_1", "sentiment": "Happy", "knowledge_source": ["FS1"], "turn_rating": "Good"}]}' http://localhost:5000/transform


Analyze Transcript:
curl -X POST -H "Content-Type: application/json" -d '{"article_url": "http://example.com", "content": [{"message": "I love this!", "agent": "agent_1"}, {"message": "Me too!", "agent": "agent_2"}]}' http://localhost:5000/analyze



Project Structure

app.py: Flask application with API endpoints.
data/: Contains the dataset.
scripts/: Data processing and analysis scripts.
data_loader.py: Data ingestion and preprocessing.
eda.py: Exploratory data analysis and visualizations (including word cloud, message length distribution, knowledge source frequency, and sentiment over turn).
sentiment.py: Sentiment analysis using DistilBERT (fine-tuned model).


static/: Stores visualization outputs (e.g., word_cloud.png, message_length_distribution.png).
templates/: For potential HTML templates.

Insights

The dataset contains chat transcripts discussing Washington Post articles.
Agents discuss sports, politics, and media, with sentiments ranging from curious to happy.
Visualizations include:
Sentiment Distribution: Shows sentiment breakdown by agent.
Message Count by Article: Displays message volume per article.
Word Cloud: Visualizes frequent words in messages.
Message Length Distribution: Analyzes message length by agent.
Knowledge Source Frequency: Shows common knowledge sources.
Sentiment Over Turn: Tracks sentiment trends within transcripts.


Sentiment analysis uses distilbert-base-uncased-finetuned-sst-2-english for accurate results.

Production Deployment
Install gunicorn and run:
pip install gunicorn
gunicorn --bind 0.0.0.0:5000 app:app
