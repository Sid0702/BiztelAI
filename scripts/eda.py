import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from wordcloud import WordCloud
from collections import Counter
import re

class EDA:
    def __init__(self, df):
        self.df = df
        self.output_dir = Path('static')

    def article_summary(self):
        """Summarize data by article."""
        article_summary = self.df.groupby('article_url').agg({
            'transcript_id': 'nunique',
            'message': 'count',
            'agent': 'nunique'
        }).reset_index()
        article_summary.columns = ['article_url', 'num_transcripts', 'num_messages', 'num_agents']
        return article_summary

    def agent_summary(self):
        """Summarize data by agent."""
        agent_summary = self.df.groupby('agent').agg({
            'message': 'count',
            'sentiment': lambda x: x.value_counts().to_dict(),
            'turn_rating': lambda x: x.value_counts().to_dict()
        }).reset_index()
        agent_summary.columns = ['agent', 'num_messages', 'sentiment_dist', 'turn_rating_dist']
        return agent_summary

    def plot_sentiment_distribution(self):
        """Plot sentiment distribution by agent."""
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.df, x='sentiment', hue='agent')
        plt.title('Sentiment Distribution by Agent')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sentiment_distribution.png')
        plt.close()

    def plot_message_count_by_article(self):
        """Plot message count by article."""
        article_counts = self.df.groupby('article_url')['message'].count().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=article_counts, x='article_url', y='message')
        plt.title('Number of Messages per Article')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'message_count_by_article.png')
        plt.close()

    def plot_word_cloud(self):
        """Generate a word cloud from message content."""
        # Combine all messages into a single string
        text = ' '.join(self.df['message'].dropna().astype(str))
        # Clean text: remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10).generate(text)
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Messages')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'word_cloud.png')
        plt.close()

    def plot_message_length_distribution(self):
        """Plot distribution of message lengths (in words)."""
        # Calculate message lengths
        self.df['message_length'] = self.df['message'].apply(lambda x: len(str(x).split()))
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x='message_length', hue='agent', bins=30)
        plt.title('Message Length Distribution by Agent')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'message_length_distribution.png')
        plt.close()

    def plot_knowledge_source_frequency(self):
        """Plot frequency of knowledge sources."""
        # Split and count knowledge sources
        knowledge_sources = self.df['knowledge_source'].str.split(',', expand=True).stack().str.strip()
        knowledge_counts = knowledge_sources.value_counts().reset_index()
        knowledge_counts.columns = ['knowledge_source', 'count']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=knowledge_counts, x='knowledge_source', y='count')
        plt.title('Knowledge Source Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'knowledge_source_frequency.png')
        plt.close()

    def plot_sentiment_over_turn(self):
        """Plot sentiment distribution over turn order."""
        # Assume turn order is the index within each transcript
        self.df['turn_order'] = self.df.groupby('transcript_id').cumcount()
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.df, x='turn_order', y='sentiment', hue='agent', style='agent', markers=True)
        plt.title('Sentiment Over Turn Order by Agent')
        plt.xlabel('Turn Order')
        plt.ylabel('Sentiment')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sentiment_over_turn.png')
        plt.close()

    def generate_visualizations(self):
        """Generate all visualizations."""
        self.output_dir.mkdir(exist_ok=True)
        self.plot_sentiment_distribution()
        self.plot_message_count_by_article()
        self.plot_word_cloud()
        self.plot_message_length_distribution()
        self.plot_knowledge_source_frequency()
        self.plot_sentiment_over_turn()
        print("Visualizations saved in static/ directory.")