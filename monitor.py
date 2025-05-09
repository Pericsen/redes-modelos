import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import os

# Import our trained classifier
from complaint_classifier import ComplaintClassifier

class SocialMediaComplaintMonitor:
    """
    A system for monitoring, classifying, and analyzing complaints on social media
    using our domain-adapted classifier
    """
    
    def __init__(self, model_path="complaint_classifier_results/model"):
        """
        Initialize the complaint monitoring system
        
        Args:
            model_path: Path to the trained complaint classifier model
        """
        # Load the pre-trained classifier
        self.classifier = ComplaintClassifier()
        self.classifier.load_model(model_path)
        
        # Initialize storage for processed complaints
        self.complaints_db = pd.DataFrame(columns=[
            'timestamp', 'platform', 'user_id', 'text', 'category', 
            'confidence', 'sentiment_score', 'urgency_score', 'is_responded'
        ])
        
        print(f"Complaint Monitor initialized with model from {model_path}")
        print(f"Available categories: {list(self.classifier.label_decoder.values())}")
    
    def preprocess_social_post(self, text):
        """
        Preprocess social media text for classification
        
        Args:
            text: Raw social media post text
            
        Returns:
            Preprocessed text
        """
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Standardize @mentions
        text = re.sub(r'@(\w+)', r'@MENTION', text)
        
        # Standardize hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_sentiment_score(self, text):
        """
        Calculate a simple sentiment score for the complaint
        
        Args:
            text: Complaint text
            
        Returns:
            Sentiment score (-1 to 1)
        """
        # This is a very simple approach - in production you would use a proper sentiment analyzer
        positive_words = ['good', 'great', 'excellent', 'thanks', 'love', 'appreciate', 'helpful', 'resolved']
        negative_words = ['bad', 'awful', 'terrible', 'horrible', 'worst', 'disappointed', 'angry', 'furious', 
                         'frustrated', 'scam', 'ripoff', 'ridiculous', 'waste', 'poor']
        
        # Count word occurrences
        text_lower = text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)
        
        # Calculate score
        total = positive_count + negative_count
        if total == 0:
            return 0
        
        return (positive_count - negative_count) / total
    
    def calculate_urgency_score(self, text):
        """
        Calculate an urgency score for the complaint
        
        Args:
            text: Complaint text
            
        Returns:
            Urgency score (0 to 1)
        """
        # Look for urgent indicators
        urgent_phrases = ['asap', 'urgent', 'immediately', 'emergency', 'now', 'help', 
                         'right now', 'quickly', 'URGENT', 'ASAP', 'need help']
        
        # Calculate base score from urgent phrases
        text_lower = text.lower()
        urgency_count = sum(text_lower.count(phrase) for phrase in urgent_phrases)
        
        # Look for temporal indicators
        time_indicators = ['today', 'tonight', 'this morning', 'this afternoon', 'this evening', 
                          'yesterday', 'tomorrow', 'next day']
        
        time_count = sum(text_lower.count(indicator) for indicator in time_indicators)
        
        # Look for excessive punctuation or capitalization (indicators of urgency)
        exclamation_count = text.count('!')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Combine into overall score
        urgency_score = min(1.0, (
            0.4 * min(1.0, urgency_count / 2) + 
            0.2 * min(1.0, time_count / 2) +
            0.2 * min(1.0, exclamation_count / 3) +
            0.2 * min(1.0, caps_ratio * 5)
        ))
        
        return urgency_score
    
    def process_social_media_post(self, post_data):
        """
        Process a social media post to detect and classify complaints
        
        Args:
            post_data: Dictionary with post metadata and content
                - 'text': The post text content
                - 'timestamp': When the post was created
                - 'platform': Social media platform (Twitter, Facebook, etc.)
                - 'user_id': User identifier
                
        Returns:
            Dictionary with classification results
        """
        # Extract text and preprocess
        text = post_data['text']
        processed_text = self.preprocess_social_post(text)
        
        # Check if it's too short to be a meaningful complaint
        if len(processed_text.split()) < 3:
            return None
        
        # Classify the text
        prediction_results = self.classifier.predict([processed_text])[0]
        
        # Calculate sentiment and urgency
        sentiment_score = self.calculate_sentiment_score(text)
        urgency_score = self.calculate_urgency_score(text)
        
        # Create result object
        result = {
            'timestamp': post_data.get('timestamp', datetime.now()),
            'platform': post_data.get('platform', 'unknown'),
            'user_id': post_data.get('user_id', 'unknown'),
            'text': text,
            'processed_text': processed_text,
            'category': prediction_results['predicted_category'],
            'confidence': prediction_results['confidence'],
            'sentiment_score': sentiment_score,
            'urgency_score': urgency_score,
            'is_responded': False
        }
        
        # Add to our database
        self.complaints_db = pd.concat([
            self.complaints_db, 
            pd.DataFrame([result])
        ], ignore_index=True)
        
        return result
        
    def process_batch(self, social_media_data):
        """
        Process a batch of social media posts
        
        Args:
            social_media_data: List of post dictionaries
                
        Returns:
            DataFrame with classification results
        """
        results = []
        for post in social_media_data:
            result = self.process_social_media_post(post)
            if result:
                results.append(result)
        
        return pd.DataFrame(results)
    
    def get_priority_complaints(self, min_confidence=0.7, min_urgency=0.5):
        """
        Get high-priority complaints that require immediate attention
        
        Args:
            min_confidence: Minimum classification confidence
            min_urgency: Minimum urgency score
                
        Returns:
            DataFrame with priority complaints
        """
        # Filter for high confidence and urgency
        priority_complaints = self.complaints_db[
            (self.complaints_db['confidence'] >= min_confidence) &
            (self.complaints_db['urgency_score'] >= min_urgency) &
            (self.complaints_db['is_responded'] == False)
        ].copy()
        
        # Sort by urgency and confidence
        priority_complaints.sort_values(
            by=['urgency_score', 'confidence'], 
            ascending=False, 
            inplace=True
        )
        
        return priority_complaints
    
    def mark_as_responded(self, complaint_indices):
        """
        Mark complaints as responded to
        
        Args:
            complaint_indices: List of indices to mark as responded
        """
        self.complaints_db.loc[complaint_indices, 'is_responded'] = True
        print(f"Marked {len(complaint_indices)} complaints as responded")
    
    def generate_dashboard_data(self, days_back=7):
        """
        Generate data for a complaints dashboard
        
        Args:
            days_back: How many days of data to include
                
        Returns:
            Dictionary with dashboard data
        """
        # Filter for recent complaints
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_df = self.complaints_db[self.complaints_db['timestamp'] >= cutoff_date].copy()
        
        # Check if we have data
        if len(recent_df) == 0:
            return {"error": "No recent complaint data available"}
        
        # 1. Category distribution
        category_counts = recent_df['category'].value_counts().to_dict()
        
        # 2. Platform distribution
        platform_counts = recent_df['platform'].value_counts().to_dict()
        
        # 3. Daily trend
        recent_df['date'] = recent_df['timestamp'].dt.date
        daily_counts = recent_df.groupby('date').size().to_dict()
        
        # 4. Response rate
        response_rate = recent_df['is_responded'].mean()
        
        # 5. Sentiment distribution
        sentiment_bins = [-1, -0.5, 0, 0.5, 1]
        sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive']
        recent_df['sentiment_category'] = pd.cut(
            recent_df['sentiment_score'], 
            bins=sentiment_bins, 
            labels=sentiment_labels
        )
        sentiment_dist = recent_df['sentiment_category'].value_counts().to_dict()
        
        # 6. Average sentiment by category
        avg_sentiment_by_category = recent_df.groupby('category')['sentiment_score'].mean().to_dict()
        
        # 7. Urgency distribution
        urgency_bins = [0, 0.3, 0.6, 1]
        urgency_labels = ['Low', 'Medium', 'High']
        recent_df['urgency_category'] = pd.cut(
            recent_df['urgency_score'], 
            bins=urgency_bins, 
            labels=urgency_labels
        )
        urgency_dist = recent_df['urgency_category'].value_counts().to_dict()
        
        # Create dashboard data dictionary
        dashboard_data = {
            'total_complaints': len(recent_df),
            'response_rate': response_rate,
            'category_distribution': category_counts,
            'platform_distribution': platform_counts,
            'daily_trend': daily_counts,
            'sentiment_distribution': sentiment_dist,
            'avg_sentiment_by_category': avg_sentiment_by_category,
            'urgency_distribution': urgency_dist,
            'most_urgent_complaint': recent_df.loc[recent_df['urgency_score'].idxmax()].to_dict() if len(recent_df) > 0 else None
        }
        
        return dashboard_data
    
    def visualize_dashboard(self, dashboard_data, output_dir="dashboard"):
        """
        Visualize dashboard data with charts
        
        Args:
            dashboard_data: Dashboard data dictionary
            output_dir: Directory to save visualization files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Category Distribution Pie Chart
        plt.figure(figsize=(10, 6))
        categories = list(dashboard_data['category_distribution'].keys())
        values = list(dashboard_data['category_distribution'].values())
        plt.pie(values, labels=categories, autopct='%1.1f%%')
        plt.title('Complaint Category Distribution')
        plt.savefig(f"{output_dir}/category_distribution.png")
        plt.close()
        
        # 2. Daily Trend Line Chart
        plt.figure(figsize=(12, 6))
        dates = list(dashboard_data['daily_trend'].keys())
        counts = list(dashboard_data['daily_trend'].values())
        plt.plot(dates, counts, marker='o')
        plt.title('Daily Complaint Volume')
        plt.xlabel('Date')
        plt.ylabel('Number of Complaints')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/daily_trend.png")
        plt.close()
        
        # 3. Sentiment Distribution Bar Chart
        plt.figure(figsize=(10, 6))
        sentiment_cats = list(dashboard_data['sentiment_distribution'].keys())
        sentiment_counts = list(dashboard_data['sentiment_distribution'].values())
        plt.bar(sentiment_cats, sentiment_counts, color=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Category')
        plt.ylabel('Count')
        plt.savefig(f"{output_dir}/sentiment_distribution.png")
        plt.close()
        
        # 4. Average Sentiment by Category
        plt.figure(figsize=(12, 6))
        cats = list(dashboard_data['avg_sentiment_by_category'].keys())
        avg_sentiments = list(dashboard_data['avg_sentiment_by_category'].values())
        plt.barh(cats, avg_sentiments)
        plt.title('Average Sentiment by Category')
        plt.xlabel('Average Sentiment (-1 to 1)')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_by_category.png")
        plt.close()
        
        print(f"Dashboard visualizations saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize the monitor
    monitor = SocialMediaComplaintMonitor(model_path="complaint_classifier_results/model")
    
    example_posts = [
        {
            'text': "@BigBank I've been charged overdraft fees 3 times this month even though I was told I had overdraft protection! This is ridiculous! #ripoff #needhelp",
            'timestamp': datetime.now() - timedelta(hours=2),
            'platform': 'Twitter',
            'user_id': 'user123'
        },
        {
            'text': "Just got my mortgage statement and there's a $500 'processing fee' that nobody told me about! @HomeLender what is this?? I need answers TODAY.",
            'timestamp': datetime.now() - timedelta(hours=5),
            'platform': 'Twitter',
            'user_id': 'user456'
        },
        {
            'text': "My credit score dropped 50 points because @CreditBureau has wrong information on my report. I've submitted disputes three times and they keep ignoring me. #frustrated #badservice",
            'timestamp': datetime.now() - timedelta(hours=8),
            'platform': 'Twitter',
            'user_id': 'user789'
        },
        {
            'text': "I love the new mobile banking app from @BigBank! So much easier to use than before.",
            'timestamp': datetime.now() - timedelta(days=1),
            'platform': 'Facebook',
            'user_id': 'user101'
        },
        {
            'text': "URGENT! My card was declined at the grocery store even though I have plenty of money in my account! @BigBank fix this immediately!",
            'timestamp': datetime.now() - timedelta(hours=1),
            'platform': 'Twitter',
            'user_id': 'user202'
        },
        {
            'text': "Been trying to reach customer service for 45 minutes. This is the worst experience I've ever had. Do better @FinanceCorp #awful",
            'timestamp': datetime.now() - timedelta(days=2),
            'platform': 'Twitter',
            'user_id': 'user303'
        },
        {
            'text': "Thanks @InvestmentFirm for helping me set up my retirement account yesterday. The advisor was very knowledgeable!",
            'timestamp': datetime.now() - timedelta(days=3),
            'platform': 'Facebook',
            'user_id': 'user404'
        },
        {
            'text': "@InsuranceCo denied my claim for water damage because they say it's 'flood damage' which isn't covered. It was a BURST PIPE not a flood! Need help ASAP before mold sets in!",
            'timestamp': datetime.now() - timedelta(days=1, hours=12),
            'platform': 'Twitter',
            'user_id': 'user505'
        },
        {
            'text': "Been a customer of @BigBank for 15 years and they can't approve my loan application? Moving all my accounts to @CompetitorBank next week.",
            'timestamp': datetime.now() - timedelta(days=4),
            'platform': 'Facebook',
            'user_id': 'user606'
        },
        {
            'text': "@PaymentApp charged me twice for the same transaction and now my account is overdrawn! Fix this TODAY!",
            'timestamp': datetime.now() - timedelta(hours=3),
            'platform': 'Twitter',
            'user_id': 'user707'
        }
    ]
    
    # Process the batch of posts
    print("Processing social media posts...")
    results_df = monitor.process_batch(example_posts)
    print(f"Processed {len(results_df)} posts")
    
    # Get priority complaints that need immediate attention
    priority_complaints = monitor.get_priority_complaints(min_confidence=0.7, min_urgency=0.5)
    print(f"\nFound {len(priority_complaints)} high-priority complaints:")
    if len(priority_complaints) > 0:
        for idx, complaint in priority_complaints.iterrows():
            print(f"- [{complaint['platform']}] {complaint['text'][:100]}... (Urgency: {complaint['urgency_score']:.2f})")
    
    # Simulate responding to some complaints
    if len(priority_complaints) > 0:
        # Respond to the top 2 most urgent complaints
        respond_indices = priority_complaints.index[:min(2, len(priority_complaints))]
        monitor.mark_as_responded(respond_indices)
    
    # Generate dashboard data for the last 7 days
    print("\nGenerating dashboard data...")
    dashboard_data = monitor.generate_dashboard_data(days_back=7)
    
    # Print some summary statistics
    print(f"\nDashboard Summary:")
    print(f"- Total complaints: {dashboard_data['total_complaints']}")
    print(f"- Response rate: {dashboard_data['response_rate']*100:.1f}%")
    print(f"- Most common category: {max(dashboard_data['category_distribution'].items(), key=lambda x: x[1])[0]}")
    print(f"- Most common platform: {max(dashboard_data['platform_distribution'].items(), key=lambda x: x[1])[0]}")
    
    # Create visualizations
    print("\nCreating dashboard visualizations...")
    monitor.visualize_dashboard(dashboard_data, output_dir="complaint_dashboard")
    
    # Export the complaints database to CSV for further analysis
    monitor.complaints_db.to_csv("complaints_database.csv", index=False)
    print("\nExported complaints database to complaints_database.csv")
    
    print("\nComplaint monitoring process complete!")