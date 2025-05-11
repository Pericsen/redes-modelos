import pandas as pd
import numpy as np
import os
import json
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Import the ComplaintClassifier from our main module
from model.social_media_complaint_classifier.classifier import ComplaintClassifier, ComplaintDataset

# 1. Data Loading and Exploration
def load_and_explore_data(data_path):
    """
    Load and explore the official complaint dataset
    
    Args:
        data_path: Path to the dataset (e.g., CFPB complaints CSV)
    
    Returns:
        Processed DataFrame
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Basic exploration
    print(f"Dataset shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Check for the text column with complaints
    text_column = "consumer_complaint_narrative"  # Adjust based on your dataset
    if text_column in df.columns:
        # Check for missing values
        missing_pct = df[text_column].isna().mean() * 100
        print(f"\nMissing values in {text_column}: {missing_pct:.2f}%")
        
        # Text length statistics
        df['text_length'] = df[text_column].fillna("").apply(len)
        print("\nText length statistics:")
        print(df['text_length'].describe())
        
        # Plot text length distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['text_length'][df['text_length'] > 0], bins=50)
        plt.title("Distribution of Complaint Text Lengths")
        plt.xlabel("Text Length (characters)")
        plt.ylabel("Count")
        plt.savefig("text_length_distribution.png")
        print("Text length distribution saved to text_length_distribution.png")
    
    # Label column exploration
    label_column = "product"  # Adjust based on your dataset
    if label_column in df.columns:
        print(f"\nLabel distribution ({label_column}):")
        label_counts = df[label_column].value_counts()
        print(label_counts)
        
        # Plot label distribution
        plt.figure(figsize=(12, 8))
        sns.barplot(x=label_counts.values, y=label_counts.index)
        plt.title(f"Distribution of {label_column}")
        plt.xlabel("Count")
        plt.savefig("label_distribution.png")
        print("Label distribution saved to label_distribution.png")
    
    return df

# 2. Text Analysis for Domain Adaptation
def analyze_text_characteristics(df, text_column):
    """
    Analyze text characteristics to inform domain adaptation strategies
    
    Args:
        df: DataFrame with complaint data
        text_column: Name of column containing complaint text
    """
    import nltk
    from nltk.corpus import stopwords
    
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Fill NA values
    texts = df[text_column].fillna("").tolist()
    
    # Sample size for detailed analysis
    sample_size = min(1000, len(texts))
    sampled_texts = np.random.choice(texts, sample_size, replace=False)
    
    # 1. Perspective Analysis (third-person vs first-person)
    first_person_count = sum(1 for text in sampled_texts if re.search(r'\b(I|my|mine|me)\b', text.lower()))
    third_person_count = sum(1 for text in sampled_texts if re.search(r'\b(they|their|them|he|she|his|her)\b', text.lower()))
    
    print("\nPerspective Analysis:")
    print(f"First-person perspective: {first_person_count / sample_size:.2%}")
    print(f"Third-person perspective: {third_person_count / sample_size:.2%}")
    
    # 2. Formality Analysis
    formal_indicators = ['therefore', 'however', 'furthermore', 'consequently', 'regarding', 'additionally']
    informal_indicators = ['like', 'so', 'just', 'pretty', 'really', 'very', 'totally']
    
    formal_count = sum(1 for text in sampled_texts if any(word in text.lower() for word in formal_indicators))
    informal_count = sum(1 for text in sampled_texts if any(word in text.lower() for word in informal_indicators))
    
    print("\nFormality Analysis:")
    print(f"Formal language indicators: {formal_count / sample_size:.2%}")
    print(f"Informal language indicators: {informal_count / sample_size:.2%}")
    
    # 3. Emotional Content
    emotion_words = ['angry', 'upset', 'frustrated', 'disappointed', 'happy', 'satisfied', 
                     'furious', 'annoyed', 'terrible', 'awful', 'horrible', 'ridiculous']
    emotion_count = sum(1 for text in sampled_texts if any(word in text.lower() for word in emotion_words))
    
    print("\nEmotional Content:")
    print(f"Texts with emotional indicators: {emotion_count / sample_size:.2%}")
    
    # 4. Special Terms/Jargon
    # This would be domain-specific, for financial complaints we might look for terms like:
    financial_terms = ['apr', 'interest rate', 'credit score', 'mortgage', 'loan', 'payment', 
                      'balance', 'fee', 'charge', 'account', 'credit card', 'statement']
    financial_term_count = sum(1 for text in sampled_texts if any(term in text.lower() for term in financial_terms))
    
    print("\nDomain-Specific Terminology:")
    print(f"Texts with financial terms: {financial_term_count / sample_size:.2%}")
    
    # 5. Common phrases that indicate third-party description
    third_party_phrases = [
        'consumer stated', 'consumer reported', 'customer indicated', 
        'complainant mentioned', 'according to the consumer'
    ]
    third_party_phrase_count = sum(1 for text in sampled_texts if any(phrase in text.lower() for phrase in third_party_phrases))
    
    print("\nThird-Party Description Indicators:")
    print(f"Texts with third-party phrases: {third_party_phrase_count / sample_size:.2%}")
    
    # These insights help inform our domain adaptation strategy

# 3. Create Social Media Style Transfer Function
def create_style_transfer_function(df, text_column, label_column, output_path="style_transfer_examples.csv"):
    """
    Create and demonstrate a style transfer function to convert third-party descriptions
    to direct consumer complaints on social media
    
    Args:
        df: DataFrame with complaint data
        text_column: Name of column containing complaint text
        label_column: Name of column containing complaint category
        output_path: Where to save examples
    """
    # Define a more comprehensive style transfer function
    def convert_to_social_media_style(text, category):
        """Convert third-party descriptions to social media complaint style"""
        if not isinstance(text, str) or len(text) < 10:
            return ""
            
        # 1. Convert third-person to first-person
        text = re.sub(r"(?i)the (?:consumer|customer|complainant|client) (?:reported|stated|indicated|mentioned|says|claimed) that (?:they|he|she|the customer)", "I", text)
        text = re.sub(r"(?i)the (?:consumer|customer|complainant|client)", "I", text)
        text = re.sub(r"(?i)consumer's|customer's|complainant's|client's", "my", text)
        text = re.sub(r"(?i)their|his|her", "my", text)
        text = re.sub(r"(?i)they were|he was|she was", "I was", text)
        
        # 2. Add emotional intensity
        emotional_enhancers = {
            "issue": ["serious issue", "major issue", "ridiculous issue"],
            "problem": ["huge problem", "nightmare", "absolute disaster"],
            "error": ["massive error", "unbelievable mistake", "ridiculous error"],
            "fee": ["outrageous fee", "ridiculous charge", "hidden fee"],
            "denied": ["flat out denied", "rudely denied", "completely denied"],
            "delay": ["endless delay", "ridiculous wait", "unacceptable delay"]
        }
        
        for word, replacements in emotional_enhancers.items():
            if word in text.lower():
                replacement = np.random.choice(replacements)
                text = re.sub(r"(?i)\b" + word + r"\b", replacement, text)
        
        # 3. Add social media elements based on category
        category_mentions = {
            "Credit card": ["@Visa", "@Mastercard", "@AmericanExpress", "@Discover", "@ChaseBank", "@BankofAmerica"],
            "Bank account": ["@ChaseBank", "@BankofAmerica", "@WellsFargo", "@CitiBank", "@CapitalOne"],
            "Mortgage": ["@QuickenLoans", "@WellsFargo", "@BankofAmerica", "@ChaseHome"],
            "Debt collection": ["@CollectionsAgency", "@DebtCollector"],
            "Credit report": ["@Equifax", "@Experian", "@TransUnion", "@CreditKarma"]
        }
        
        # Add relevant @mention based on category if available
        if category in category_mentions:
            possible_mentions = category_mentions[category]
            mentioned_company = np.random.choice(possible_mentions)
            
            # 50% chance to add at beginning, 50% chance at end
            if np.random.random() < 0.5:
                text = f"{mentioned_company} {text}"
            else:
                text = f"{text} {mentioned_company}"
        
        # 4. Add hashtags related to complaints
        hashtags = ["#customerservice", "#complaint", "#frustrated", "#help", "#terrible", 
                   "#needhelp", "#badservice", "#unhappy", "#ripoff", "#scam", "#fail"]
        
        # Select 1-3 random hashtags
        selected_hashtags = " ".join(np.random.choice(hashtags, size=np.random.randint(1, 4), replace=False))
        
        # Add hashtags at the end
        text = f"{text} {selected_hashtags}"
        
        # 5. Shorten text for social media
        if len(text) > 280:  # Twitter-like length constraint
            sentences = re.split(r'(?<=[.!?])\s+', text)
            shortened_text = ""
            for sentence in sentences:
                if len(shortened_text) + len(sentence) <= 270:  # Leave room for hashtags
                    shortened_text += sentence + " "
                else:
                    break
            text = shortened_text.strip()
        
        # 6. Make more conversational/informal
        text = text.replace("was not", "wasn't")
        text = text.replace("did not", "didn't")
        text = text.replace("cannot", "can't")
        text = text.replace("will not", "won't")
        text = re.sub(r"(?i)I am", "I'm", text)
        text = re.sub(r"(?i)they are", "they're", text)
        
        # 7. Add emphasis with capitalization (randomly)
        if np.random.random() < 0.3:
            emphasis_words = ["never", "always", "terrible", "horrible", "worst", "awful"]
            for word in emphasis_words:
                if word in text.lower():
                    text = re.sub(r"(?i)\b" + word + r"\b", word.upper(), text)
        
        return text
    
    # Apply to a sample of the data
    sample_size = min(500, len(df))
    sample_df = df.sample(sample_size)
    
    # Create examples of transformed text
    examples = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Creating style transfers"):
        original_text = row[text_column]
        if isinstance(original_text, str) and len(original_text) > 10:
            category = row[label_column] if label_column in row else "Unknown"
            social_media_text = convert_to_social_media_style(original_text, category)
            
            examples.append({
                "original_text": original_text,
                "social_media_text": social_media_text,
                "category": category
            })
    
    # Save examples
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(output_path, index=False)
    print(f"Style transfer examples saved to {output_path}")
    
    # Show a few examples
    print("\nExample Style Transfers:")
    for i in range(min(5, len(examples))):
        print(f"\nOriginal: {examples[i]['original_text'][:200]}...")
        print(f"Social Media: {examples[i]['social_media_text']}")
        print(f"Category: {examples[i]['category']}")
    
    return convert_to_social_media_style

# 4. Enhanced Data Augmentation
def augment_training_data(df, text_column, label_column, style_transfer_func, augmentation_factor=1.0):
    """
    Create an augmented training dataset that includes both original and 
    style-transferred examples
    
    Args:
        df: DataFrame with complaint data
        text_column: Name of column containing complaint text
        label_column: Name of column containing complaint category
        style_transfer_func: Function to convert official text to social media style
        augmentation_factor: How many augmented examples to create (1.0 = same size as original)
    
    Returns:
        Augmented DataFrame
    """
    print(f"Augmenting dataset with social media style text (factor: {augmentation_factor})...")
    
    # Filter for valid text entries
    valid_df = df[df[text_column].notna() & (df[text_column].str.len() > 10)].copy()
    
    # Determine how many samples to augment
    augment_size = int(len(valid_df) * augmentation_factor)
    augment_df = valid_df.sample(augment_size, replace=(augment_size > len(valid_df)))
    
    # Create augmented examples
    augmented_texts = []
    augmented_labels = []
    
    for _, row in tqdm(augment_df.iterrows(), total=len(augment_df), desc="Generating augmented examples"):
        original_text = row[text_column]
        category = row[label_column]
        
        # Create social media style version
        social_media_text = style_transfer_func(original_text, category)
        
        augmented_texts.append(social_media_text)
        augmented_labels.append(category)
    
    # Create new DataFrame with augmented data
    augmented_df = pd.DataFrame({
        text_column: augmented_texts,
        label_column: augmented_labels,
        'is_augmented': True
    })
    
    # Add is_augmented column to original data
    valid_df['is_augmented'] = False
    
    # Combine original and augmented data
    combined_df = pd.concat([valid_df, augmented_df], ignore_index=True)
    
    print(f"Original data size: {len(valid_df)}")
    print(f"Augmented data size: {len(augmented_df)}")
    print(f"Combined data size: {len(combined_df)}")
    
    return combined_df

# 5. Train with Domain Adaptation
def train_with_domain_adaptation(combined_df, text_column, label_column, output_dir="models/complaint_classifier"):
    """
    Train a model with domain adaptation techniques
    
    Args:
        combined_df: DataFrame with original and augmented data
        text_column: Name of column containing complaint text
        label_column: Name of column containing complaint category
        output_dir: Where to save the model
    
    Returns:
        Trained classifier
    """
    # Initialize classifier
    num_labels = combined_df[label_column].nunique()
    classifier = ComplaintClassifier(model_name="distilbert-base-uncased", num_labels=num_labels)
    
    # Split data with stratification
    train_df, val_df = train_test_split(
        combined_df, 
        test_size=0.2, 
        stratify=combined_df[[label_column, 'is_augmented']],
        random_state=42
    )
    
    print(f"Training data size: {len(train_df)}")
    print(f"Validation data size: {len(val_df)}")
    
    # Create label encodings
    label_encoder = {label: i for i, label in enumerate(combined_df[label_column].unique())}
    label_decoder = {i: label for label, i in label_encoder.items()}
    
    # Create datasets
    def create_dataset(df):
        texts = df[text_column].fillna("").tolist()
        labels = df[label_column].map(label_encoder).tolist()
        
        dataset = ComplaintDataset(
            texts=texts,
            labels=labels,
            tokenizer=classifier.tokenizer,
            max_length=classifier.max_length
        )
        return dataset
    
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    
    # Set label mappings for the classifier
    classifier.label_encoder = label_encoder
    classifier.label_decoder = label_decoder
    
    # Train the model
    classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=16,
        epochs=3,
        learning_rate=2e-5
    )
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    classifier.save_model(output_dir)
    
    return classifier

# 6. Evaluate on Social Media Examples
def evaluate_on_social_media(classifier, social_media_examples_path, output_path="social_media_evaluation.json"):
    """
    Evaluate the trained model on real social media examples
    
    Args:
        classifier: Trained ComplaintClassifier
        social_media_examples_path: Path to labeled social media examples CSV
        output_path: Where to save evaluation results
    """
    # Load social media examples (if available)
    try:
        social_media_df = pd.read_csv(social_media_examples_path)
        print(f"Loaded {len(social_media_df)} social media examples")
        
        # Get texts and true labels
        texts = social_media_df['text'].tolist()
        true_labels = social_media_df['category'].tolist()
        
        # Make predictions
        results = classifier.predict(texts)
        
        # Extract predictions
        pred_labels = [r['predicted_category'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        # Calculate accuracy
        correct = sum(1 for true, pred in zip(true_labels, pred_labels) if true == pred)
        accuracy = correct / len(true_labels)
        
        print(f"Accuracy on social media examples: {accuracy:.4f}")
        
        # Save detailed results
        detailed_results = []
        for i, (text, true, pred, conf) in enumerate(zip(texts, true_labels, pred_labels, confidences)):
            detailed_results.append({
                "id": i,
                "text": text,
                "true_category": true,
                "predicted_category": pred,
                "confidence": conf,
                "correct": true == pred
            })
        
        with open(output_path, 'w') as f:
            json.dump({
                "accuracy": accuracy,
                "results": detailed_results
            }, f, indent=2)
        
        print(f"Evaluation results saved to {output_path}")
        
    except FileNotFoundError:
        print(f"Warning: {social_media_examples_path} not found. Skipping social media evaluation.")

# 7. Generate Confidence Calibration
def generate_confidence_calibration(classifier, val_dataset, output_path="confidence_calibration.png"):
    """
    Generate a confidence calibration curve to analyze model reliability
    
    Args:
        classifier: Trained ComplaintClassifier
        val_dataset: Validation dataset
        output_path: Where to save the calibration curve
    """
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    
    # Get predictions and confidences
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    all_preds = []
    all_labels = []
    all_confidences = []
    
    classifier.model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Calculating calibration"):
            input_ids = batch['input_ids'].to(classifier.device)
            attention_mask = batch['attention_mask'].to(classifier.device)
            labels = batch['labels'].to(classifier.device)
            
            outputs = classifier.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to binary problem (correct/incorrect)
    binary_true = [1 if pred == label else 0 for pred, label in zip(all_preds, all_labels)]
    
    # Generate calibration curve
    prob_true, prob_pred = calibration_curve(binary_true, all_confidences, n_bins=10)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(output_path)
    
    print(f"Calibration curve saved to {output_path}")

# 8. End-to-End Pipeline
def run_end_to_end_pipeline(data_path, 
                           social_media_examples_path=None, 
                           output_dir="complaint_classifier"):
    """
    Run the full pipeline from data loading to model evaluation
    
    Args:
        data_path: Path to the official complaint dataset
        social_media_examples_path: Path to labeled social media examples (optional)
        output_dir: Directory to save models and results
    """
    print("Starting end-to-end pipeline...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and explore data
    df = load_and_explore_data(data_path)
    
    text_column = "consumer_complaint_narrative"  # Adjust based on your dataset
    label_column = "product"  # Adjust based on your dataset
    
    # 2. Analyze text characteristics
    analyze_text_characteristics(df, text_column)
    
    # 3. Create style transfer function
    style_transfer_func = create_style_transfer_function(
        df, 
        text_column, 
        label_column,
        output_path=f"{output_dir}/style_transfer_examples.csv"
    )
    
    # 4. Augment training data
    combined_df = augment_training_data(
        df, 
        text_column, 
        label_column, 
        style_transfer_func,
        augmentation_factor=0.5  # Create additional 50% of data
    )
    
    # 5. Train with domain adaptation
    classifier = train_with_domain_adaptation(
        combined_df,
        text_column,
        label_column,
        output_dir=f"{output_dir}/model"
    )
    
    # 6. Evaluate on social media examples (if available)
    if social_media_examples_path:
        evaluate_on_social_media(
            classifier,
            social_media_examples_path,
            output_path=f"{output_dir}/social_media_evaluation.json"
        )
    
    # 7. Test on example social media posts
    test_examples = [
        "@BankXYZ seriously? Another overdraft fee when my deposit was pending? This is robbery! #frustrated #needhelp",
        "Just got my credit report and there's an account I NEVER opened! Identity theft is no joke people! @Experian #furious",
        "Been trying to refinance my mortgage for 3 months and @HomeLender keeps 'losing' my paperwork. Worst. Service. Ever.",
        "Credit card company just raised my APR for NO REASON. 29.99%?! That should be illegal! #ripoff #badservice",
        "Bank froze my account and won't tell me why. 3 hours on hold and still no answers. I need access to MY money!!!"
    ]
    
    results = classifier.predict(test_examples)
    
    print("\nSocial Media Test Examples:")
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("---")
    
    print(f"\nPipeline complete! Model and results saved to {output_dir}")

if __name__ == "__main__":
    # Replace with your data paths
    data_path = "consumer_complaints.csv"  # Official CFPB complaints dataset
    social_media_examples_path = "social_media_complaints.csv"  # Optional labeled social media examples
    
    run_end_to_end_pipeline(
        data_path,
        social_media_examples_path,
        output_dir="complaint_classifier_results"
    )