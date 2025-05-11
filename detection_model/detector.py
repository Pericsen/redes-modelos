import pandas as pd
import numpy as np
import re
import os
import dill as pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK required resources
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK Punkt tokenizer already downloaded")
except LookupError:
    print("Downloading NLTK Punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
    print("NLTK Stopwords already downloaded")
except LookupError:
    print("Downloading NLTK Stopwords...")
    nltk.download('stopwords')

# Load Spanish stopwords
spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))
    
# Define feature extractors
class ComplaintFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features that indicate valid complaints"""
    
    def __init__(self):
        self.location_words = [
            'calle', 'avenida', 'plaza', 'barrio', 'esquina', 'cuadra', 
            'vereda', 'manzana', 'casa', 'edificio', 'parque', 'puente',
            'zona', 'área', 'sector', 'distrito', 'pasaje', 'boulevard',
            'paseo', 'camino', 'carretera', 'autopista', 'ruta'
        ]
        self.infrastructure_words = [
            'poste', 'luz', 'agua', 'cloaca', 'alcantarilla', 'bache',
            'semáforo', 'señal', 'tránsito', 'basura', 'contenedor',
            'árbol', 'banco', 'asiento', 'parada', 'vereda', 'acera',
            'asfalto', 'pavimento', 'drenaje', 'desagüe', 'alumbrado',
            'luminaria', 'cámara', 'seguridad', 'incendio', 'escape',
            'fuga', 'pérdida', 'rotura', 'ruptura', 'caída', 'derrumbe'
        ]
        self.official_words = [
            'reclamo', 'expediente', 'exp', 'solicitud', 'número', 
            'trámite', 'registro', 'denuncia', 'municipalidad', 'queja',
            'referencia', 'caso', 'incidente', 'gestión', 'ticket', 
            'soporte', 'atención', 'servicio', 'seguimiento'
        ]
        self.action_verbs = [
            'arreglar', 'solucionar', 'reparar', 'reemplazar', 'instalar',
            'remover', 'limpiar', 'atender', 'revisar', 'inspeccionar', 
            'verificar', 'gestionar', 'modificar', 'actualizar', 'cambiar'
        ]
        
        # Named entity patterns (simple regex approach instead of spaCy NER)
        self.location_patterns = [
            r'\b(?:en|sobre|cerca de|junto a|frente a|detrás de)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bcalle\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bavenida\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bplaza\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+',
            r'\bbarrio\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+'
        ]
        
        # Compile regex patterns for efficiency
        self.phone_pattern = re.compile(r'\b(?:\+?54)?(?:11|15)?[0-9]{8,10}\b')
        self.expedition_pattern = re.compile(r'\b(?:exp(?:ediente)?\.?\s?(?:n[°º]?\.?\s?)?[0-9-]+\/[0-9]{4})\b', re.IGNORECASE)
        self.incident_pattern = re.compile(r'\b(?:incidente|caso|ticket|reclamo)\s?(?:n[°º]?\.?\s?)?[0-9-]+\b', re.IGNORECASE)
        self.date_pattern = re.compile(r'\b(?:(?:0?[1-9]|[12][0-9]|3[01])[\/-](?:0?[1-9]|1[0-2])[\/-][0-9]{4}|(?:0?[1-9]|1[0-2])[\/-](?:0?[1-9]|[12][0-9]|3[01])[\/-][0-9]{4})\b')
        self.location_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.location_patterns]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        """Extract complaint features from text"""
        features = np.zeros((len(X), 13))
        
        for i, text in enumerate(X):
            if isinstance(text, str):
                text_lower = text.lower()
                # Use NLTK for tokenization instead of spaCy
                tokens = nltk.word_tokenize(text_lower, language='spanish')
                
                # Length features
                features[i, 0] = len(text)
                features[i, 1] = len(tokens)
                
                # Count types of words
                location_count = sum(1 for word in tokens if word in self.location_words)
                infra_count = sum(1 for word in tokens if word in self.infrastructure_words)
                official_count = sum(1 for word in tokens if word in self.official_words)
                action_count = sum(1 for word in tokens if word in self.action_verbs)
                
                features[i, 2] = location_count
                features[i, 3] = infra_count  
                features[i, 4] = official_count
                features[i, 5] = action_count
                
                # Simple location named entity detection using patterns
                location_entities = 0
                for pattern in self.location_patterns:
                    location_entities += len(re.findall(pattern, text))
                features[i, 6] = location_entities
                
                # Contains specific patterns
                features[i, 7] = 1 if re.search(self.phone_pattern, text) else 0
                features[i, 8] = 1 if re.search(self.expedition_pattern, text) else 0
                features[i, 9] = 1 if re.search(self.incident_pattern, text) else 0
                features[i, 10] = 1 if re.search(self.date_pattern, text) else 0
                
                # Contains question vs imperative sentence
                features[i, 11] = 1 if '?' in text else 0
                features[i, 12] = 1 if '!' in text else 0
        
        return features

def preprocess_text(text):
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove extra spaces
    text = text.lower().strip()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Replace hashtags with just the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove mentions but keep names for context
    text = re.sub(r'@(\w+)', r'\1', text)
    
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text

def build_complaint_classifier(model_type="xgboost"):
    """Build the classification pipeline"""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    # Define feature extraction pipeline
    features = FeatureUnion([
        ('text_features', Pipeline([
            ('tfidf', TfidfVectorizer(
                preprocessor=preprocess_text,
                min_df=3, 
                max_df=0.8, 
                ngram_range=(1, 2),
                use_idf=True,
                smooth_idf=True,
                stop_words=list(spanish_stopwords)
            ))
        ])),
        ('complaint_features', ComplaintFeatureExtractor())
    ])

    """Ver de usar optimizacion bayesiana para los hiperparametros"""
    
    classifier = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # Full pipeline
    pipeline = Pipeline([
        ('features', features),
        ('classifier', classifier)
    ])
    
    return pipeline

def load_and_prepare_data(comments_file, labeled_data_file=None):
    """Load comments and any labeled data available"""

    # Load comments
    df = pd.read_csv(comments_file)
    df = df.dropna(subset=['comment_text'])
    df = df.drop_duplicates(subset=['comment_text'])
    df = df.reset_index(drop=True)
    
    # Make sure we have the right column
    if 'comment_text' in df.columns:
        comment_col = 'comment_text'
    elif 'message' in df.columns:
        comment_col = 'message'
    else:
        raise ValueError("Could not find comments column in data")
    
    # Extract the raw text
    comments = df[comment_col].tolist()
    
    # If we have labeled data, use it for training
    if labeled_data_file and os.path.exists(labeled_data_file):
        labeled_df = pd.read_csv(labeled_data_file)

        labeled_df = labeled_df.dropna(subset=['text', 'is_valid_complaint'])
        
        # Make sure it has the expected columns
        required_cols = ['text', 'is_valid_complaint']
        if not all(col in labeled_df.columns for col in required_cols):
            raise ValueError(f"Labeled data must have columns: {required_cols}")
        
        return df, labeled_df
    else:
        # If no labeled data, return just Facebook data
        return df, None

def train_model_with_labeled_data(labeled_df, model_path="complaint_classifier_model.pkl"):
    """Train the model with labeled data"""
    # Split data
    X = labeled_df['text']
    y = labeled_df['is_valid_complaint']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build and train model
    pipeline = build_complaint_classifier()
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save model
    with open(model_path, 'wb') as file:
        pickle.dump(pipeline, file)
    
    print(f"\nModel saved to {model_path}")
    return pipeline

def predict_and_analyze(df, model, comment_col='comment_text', threshold=0.7):
    """Make predictions on Facebook comments"""
    # Extract comments
    comments = df[comment_col].tolist()
    
    # Get probabilities
    probas = model.predict_proba(comments)
    
    # Add predictions to dataframe
    df['complaint_probability'] = probas[:, 1]
    df['is_valid_complaint'] = (probas[:, 1] >= threshold).astype(int)
    
    # Show some examples of valid complaints
    print("\nTop 5 Most Likely Valid Complaints:")
    valid_complaints = df[df['is_valid_complaint'] == 1].sort_values(
        'complaint_probability', ascending=False
    ).head(5)
    
    for i, (_, row) in enumerate(valid_complaints.iterrows()):
        print(f"\n{i+1}. Probability: {row['complaint_probability']:.3f}")
        print(f"Comment: {row[comment_col]}")
    
    # Show some examples of non-valid complaints/comments
    print("\nTop 5 Most Likely Non-Valid Comments:")
    non_valid = df[df['is_valid_complaint'] == 0].sort_values(
        'complaint_probability', ascending=True
    ).head(5)
    
    for i, (_, row) in enumerate(non_valid.iterrows()):
        print(f"\n{i+1}. Probability: {1 - row['complaint_probability']:.3f} (not valid)")
        print(f"Comment: {row[comment_col]}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'classified_comments_{timestamp}.csv'
    df.to_csv(result_file, index=False)
    print(f"\nResults saved to {result_file}")
    
    # Summary stats
    valid_count = df['is_valid_complaint'].sum()
    total = len(df)
    print(f"\nSummary: Found {valid_count} valid complaints out of {total} comments ({valid_count/total*100:.1f}%)")
    
    return df

# Main execution functions
def bootstrap_label_data(df, sample_size=300, output_file="complaint_labels_template.xlsx"):
    """Create a template file for manual labeling"""
    # If we don't have enough data, use what we have
    sample_size = min(sample_size, len(df))
    
    # Get comment column
    if 'comment_text' in df.columns:
        comment_col = 'comment_text'
    elif 'message' in df.columns:
        comment_col = 'message'
    else:
        raise ValueError("Could not find comments column in Facebook data")
    
    # Take a stratified sample based on basic heuristics
    df['length'] = df[comment_col].apply(lambda x: len(str(x)) if isinstance(x, str) else 0)
    df['has_numbers'] = df[comment_col].apply(
        lambda x: 1 if isinstance(x, str) and bool(re.search(r'\d', x)) else 0
    )
    
    # Define a preliminary strata using length and number presence
    df['strata'] = df['length'].apply(
        lambda x: 0 if x < 50 else (1 if x < 150 else 2)
    ) * 2 + df['has_numbers']
    
    # Sample from each strata
    samples = []
    for strata in df['strata'].unique():
        strata_df = df[df['strata'] == strata]
        strata_size = max(1, int(sample_size * len(strata_df) / len(df)))
        
        if len(strata_df) > strata_size:
            samples.append(strata_df.sample(strata_size, random_state=42))
        else:
            samples.append(strata_df)
    
    # Combine samples and prepare labeling file
    sample_df = pd.concat(samples)
    
    # Make sure we don't exceed desired sample size
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(sample_size, random_state=42)
    
    # Create labeling template
    labeling_df = pd.DataFrame({
        'text': sample_df[comment_col],
        'is_valid_complaint': ''  # This will be filled by the person labeling
    })

    labeling_df.drop_duplicates(subset=['text'], inplace=True)
    labeling_df.reset_index(drop=True, inplace=True)
    
    # Save template
    labeling_df.to_excel(output_file, index=False)
    print(f"Created labeling template with {len(labeling_df)} samples at {output_file}")
    print("Please fill the 'is_valid_complaint' column with 1 (valid) or 0 (not valid)")
    
    return labeling_df

def run_active_learning(comments_file, labeled_data_file=None, model_path=None):
    """Main function to run the active learning process"""
    # Load data
    df, labeled_df = load_and_prepare_data(comments_file, labeled_data_file)
    
    # Determine which comment column to use
    if 'comment_text' in df.columns:
        comment_col = 'comment_text'
    elif 'message' in df.columns:
        comment_col = 'message'
    else:
        raise ValueError("Could not find comments column in Facebook data")
    
    # If no labeled data, create a template for labeling
    if labeled_df is None:
        print("\nNo labeled data found. Creating a template for manual labeling...")
        bootstrap_label_data(df)
        print("\nPlease label the data and run this script again with the labeled file.")
        return
    
    # If we have labeled data, train the model
    print("\nTraining model with labeled data...")
    model = train_model_with_labeled_data(labeled_df, model_path)
    
    # Make predictions on the full dataset
    print("\nMaking predictions on all comments...")
    result_df = predict_and_analyze(df, model, comment_col)
    
    print("\nProcess complete. You can now:")
    print("1. Review the output file with predictions")
    print("2. Add more labeled examples to improve the model")
    print("3. Adjust the threshold if needed (current: 0.5)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Social Media Complaint Classifier')
    parser.add_argument('--raw', help='Path to the social media comments CSV file')
    parser.add_argument('--labeled', help='Path to labeled data CSV file (if available)')
    parser.add_argument('--model', help='Path to save/load the model', default='complaint_classifier_model.pkl')
    
    args = parser.parse_args()
    
    run_active_learning(args.raw, args.labeled, args.model)