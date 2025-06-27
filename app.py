from flask import Flask, jsonify, request, send_from_directory
from scripts.data_loader import DataLoader
from scripts.eda import EDA
from scripts.sentiment import SentimentAnalyzer
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)
data_loader = DataLoader('data/BiztelAI_DS_Dataset_V1.json')
data_loader.load_data()
df = data_loader.preprocess_data()
eda = EDA(df)
sentiment_analyzer = SentimentAnalyzer()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Welcome to BiztelAI Data Science Assignment API',
        'endpoints': {
            '/summary': 'GET - Fetch dataset summaries and visualization paths',
            '/transform': 'POST - Preprocess a new transcript',
            '/analyze': 'POST - Analyze a transcript for insights'
        }
    })

@app.route('/favicon.ico', methods=['GET'])
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/summary', methods=['GET'])
def get_summary():
    try:
        # Generate visualizations
        eda.generate_visualizations()
        
        # Get summaries
        article_summary = eda.article_summary().to_dict(orient='records')
        agent_summary = eda.agent_summary().to_dict(orient='records')
        
        # List visualization files
        viz_files = [str(f) for f in Path('static').glob('*.png')]
        
        return jsonify({
            'article_summary': article_summary,
            'agent_summary': agent_summary,
            'visualizations': viz_files
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transform', methods=['POST'])
def transform_data():
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Create a temporary DataFrame
        records = []
        for msg in data['content']:
            records.append({
                'transcript_id': data.get('transcript_id', 'temp'),
                'article_url': data.get('article_url', ''),
                'config': data.get('config', ''),
                'message': msg.get('message', ''),
                'agent': msg.get('agent', ''),
                'sentiment': msg.get('sentiment', 'Unknown'),
                'knowledge_source': ','.join(msg.get('knowledge_source', [])),
                'turn_rating': msg.get('turn_rating', 'Unknown'),
                'conversation_rating': data.get('conversation_rating', {}).get(msg.get('agent', ''), 'Unknown')
            })
        df_temp = pd.DataFrame(records)
        df_temp.fillna({'message': '', 'sentiment': 'Unknown', 'knowledge_source': '', 'turn_rating': 'Unknown', 'conversation_rating': 'Unknown'}, inplace=True)
        df_temp.drop_duplicates(inplace=True)
        
        return jsonify(df_temp.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    try:
        data = request.get_json()
        result = sentiment_analyzer.analyze_transcript(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)