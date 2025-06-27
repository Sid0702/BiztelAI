import pandas as pd
import numpy as np
import json
from pathlib import Path

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.df = None

    def load_data(self):
        """Load JSON dataset."""
        try:
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        """Preprocess the dataset into a DataFrame."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Flatten the JSON into a list of messages
        records = []
        for transcript_id, transcript in self.data.items():
            article_url = transcript.get('article_url', '')
            config = transcript.get('config', '')
            for msg in transcript.get('content', []):
                records.append({
                    'transcript_id': transcript_id,
                    'article_url': article_url,
                    'config': config,
                    'message': msg.get('message', ''),
                    'agent': msg.get('agent', ''),
                    'sentiment': msg.get('sentiment', ''),
                    'knowledge_source': ','.join(msg.get('knowledge_source', [])),
                    'turn_rating': msg.get('turn_rating', ''),
                    'conversation_rating': transcript.get('conversation_rating', {}).get(msg.get('agent', ''), '')
                })

        # Create DataFrame
        self.df = pd.DataFrame(records)

        # Handle missing values
        self.df.fillna({'message': '', 'sentiment': 'Unknown', 'knowledge_source': '', 'turn_rating': 'Unknown', 'conversation_rating': 'Unknown'}, inplace=True)

        # Remove duplicates
        self.df.drop_duplicates(inplace=True)

        # Convert data types
        self.df['transcript_id'] = self.df['transcript_id'].astype(str)
        self.df['article_url'] = self.df['article_url'].astype(str)
        self.df['config'] = self.df['config'].astype('category')
        self.df['agent'] = self.df['agent'].astype('category')
        self.df['sentiment'] = self.df['sentiment'].astype('category')
        self.df['turn_rating'] = self.df['turn_rating'].astype('category')
        self.df['conversation_rating'] = self.df['conversation_rating'].astype('category')

        print("Data preprocessed successfully.")
        return self.df

    def get_dataframe(self):
        """Return the preprocessed DataFrame."""
        if self.df is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        return self.df