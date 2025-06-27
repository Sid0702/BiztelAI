from transformers import pipeline
import pandas as pd

class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

    def analyze_transcript(self, transcript_data):
        """Analyze a single transcript."""
        if not transcript_data or 'content' not in transcript_data or 'article_url' not in transcript_data:
            raise ValueError("Invalid transcript data.")

        article_url = transcript_data['article_url']
        messages = transcript_data['content']

        # Convert to DataFrame
        df = pd.DataFrame(messages)

        # Count messages per agent
        message_counts = df['agent'].value_counts().to_dict()

        # Analyze sentiments
        agent_1_msgs = df[df['agent'] == 'agent_1']['message'].tolist()
        agent_2_msgs = df[df['agent'] == 'agent_2']['message'].tolist()

        # Handle empty message lists
        agent_1_sentiments = [self.sentiment_pipeline(msg)[0] for msg in agent_1_msgs if msg.strip()] if agent_1_msgs else []
        agent_2_sentiments = [self.sentiment_pipeline(msg)[0] for msg in agent_2_msgs if msg.strip()] if agent_2_msgs else []

        # Aggregate sentiments (default to 'NEUTRAL' if no messages)
        agent_1_overall = max(set([s['label'] for s in agent_1_sentiments]), key=[s['label'] for s in agent_1_sentiments].count) if agent_1_sentiments else 'NEUTRAL'
        agent_2_overall = max(set([s['label'] for s in agent_2_sentiments]), key=[s['label'] for s in agent_2_sentiments].count) if agent_2_sentiments else 'NEUTRAL'

        return {
            'article_url': article_url,
            'message_count': {
                'agent_1': message_counts.get('agent_1', 0),
                'agent_2': message_counts.get('agent_2', 0)
            },
            'sentiment': {
                'agent_1': agent_1_overall,
                'agent_2': agent_2_overall
            }
        }