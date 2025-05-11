import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import re
import nltk
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Pre-download NLTK data
try:
    nltk.data.find('corpora/stopwords')
    print("Stopwords already downloaded")
except LookupError:
    print("Downloading stopwords...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt already downloaded")
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt', quiet=True)

# Load stopwords once
from nltk.corpus import stopwords
spanish_stop = stopwords.words('spanish')

class ComplaintClassifier:
    def __init__(self, model_name="VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis", num_labels=5):
        """
        Initialize the complaint classifier with a pre-trained model
        
        Args:
            model_name: The name of the pre-trained model to use
            num_labels: Number of complaint categories to classify
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.max_length = 256

    def load_and_preprocess_data(self, data_path, text_column, label_column, test_size=0.2):
        """
        Load and preprocess the official complaint dataset
        
        Args:
            data_path: Path to the dataset file (CSV)
            text_column: Name of column containing complaint text
            label_column: Name of column containing complaint category
            test_size: Fraction of data to use for testing
        
        Returns:
            Processed train and validation datasets
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Basic cleaning
        df[text_column] = df[text_column].fillna("")
        df[text_column] = df[text_column].apply(self._clean_text)
        
        # Convert labels to numerical values
        self.label_encoder = {label: i for i, label in enumerate(df[label_column].unique())}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}
        df['label_id'] = df[label_column].map(self.label_encoder)
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['label_id'], random_state=42)
        
        # Create PyTorch datasets
        train_dataset = ComplaintDataset(
            texts=train_df[text_column].tolist(),
            labels=train_df['label_id'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        val_dataset = ComplaintDataset(
            texts=val_df[text_column].tolist(),
            labels=val_df['label_id'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        print(f"Data loaded with {len(train_df)} training samples and {len(val_df)} validation samples")
        print(f"Label mapping: {self.label_encoder}")
        
        return train_dataset, val_dataset
    
    def _clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def apply_domain_adaptation(self, train_dataset, social_media_examples=None):
        """
        Apply domain adaptation techniques to help bridge the gap between
        third-party descriptions and direct consumer complaints
        
        Args:
            train_dataset: The original training dataset
            social_media_examples: Optional small dataset of real social media complaints
                                  to use for adaptation
        
        Returns:
            Enhanced training dataset
        """
        # If we have real social media examples, we can use them for fine-tuning
        if social_media_examples:
            # Combine the datasets, giving more weight to social media examples
            # Implementation depends on format of social_media_examples
            pass
        
        # Style transfer function (simplified version)
        # This would ideally be a more sophisticated model that transforms
        # third-party formal language into direct consumer language
        def convert_to_consumer_style(text):
            """Convert third-party descriptions to more consumer-like language"""
            # Remove formalities and passive voice indicators
            text = re.sub(r"el vecino (solicita)", "Yo solicito", text)
            text = re.sub(r"la vecina(solicita|indica)", "Yo solicito", text) 
            text = re.sub(r"el vecino (reclama)", "Yo reclamo", text)
            text = re.sub(r"la vecina(reclama|indica)", "Yo reclamo", text) 
            text = re.sub(r"vecino (reclama|reitera|solicita)", "solicito", text)
            text = re.sub(r"vecina (reclama|reitera|solicita)", "solicito", text)
                
            
            return text
        
        # Data augmentation could be applied here
        # For a full implementation, we would create additional training examples
        
        return train_dataset
    
    def train(self, train_dataset, val_dataset, batch_size=32, epochs=3, learning_rate=2e-5):
        """Optimized training with speed improvements"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

        # Freeze early layers to speed up training
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False

        # Only fine-tune the last 4 transformer layers
        for i in range(8):
            for param in self.model.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        # Optimizer with correct learning rate
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Setup mixed precision training
        scaler = GradScaler(device="cuda")

        # Calculate steps with gradient accumulation
        accumulation_steps = 4
        total_steps = (len(train_loader) // accumulation_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            print(f"Starting epoch {epoch+1}/{epochs}")
            running_loss = 0.0

            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass with mixed precision
                with autocast(device_type="cuda"):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                running_loss += loss.item() * accumulation_steps

                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

            # Print epoch results
            avg_train_loss = running_loss / len(train_loader)
            print(f"Average training loss: {avg_train_loss:.4f}")

            # Evaluate on validation set
            val_report = self.evaluate(val_loader)
            print(f"Validation Report:\n{val_report}")
    
    def evaluate(self, data_loader):
        """Evaluate the model on the provided data loader"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # Convert numeric predictions back to original category labels
        decoded_preds = [self.label_decoder[p] for p in all_preds]
        decoded_labels = [self.label_decoder[l] for l in all_labels]
        
        # Calculate classification report
        report = classification_report(decoded_labels, decoded_preds)
        
        return report
    
    def predict(self, texts):
        """
        Predict the complaint category for new texts
        
        Args:
            texts: List of text content to classify
        
        Returns:
            Predicted categories and confidence scores
        """
        self.model.eval()
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Tokenize
        encoded = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get predicted class and confidence
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidences, predictions = torch.max(probs, dim=1)
        
        # Convert to original category labels
        results = []
        for i, (pred, conf) in enumerate(zip(predictions.cpu().numpy(), confidences.cpu().numpy())):
            category = self.label_decoder[pred]
            results.append({
                "text": texts[i],
                "predicted_category": category,
                "confidence": float(conf)
            })
        
        return results
    
    def save_model(self, output_dir):
        """Save the trained model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            import json
            json.dump({
                "label_encoder": self.label_encoder,
                "label_decoder": self.label_decoder
            }, f)
        
        print(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir):
        """Load a trained model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load label mappings
        with open(f"{model_dir}/label_mappings.json", "r") as f:
            import json
            mappings = json.load(f)
            self.label_encoder = mappings["label_encoder"]
            self.label_decoder = mappings["label_decoder"]
        
        print(f"Model loaded from {model_dir}")


class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Dataset for complaint classification
        
        Args:
            texts: List of text content
            labels: List of corresponding labels
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding['labels'] = torch.tensor(label)
        
        return encoding


# Example usage
def main():
    # Initialize the classifier
    classifier = ComplaintClassifier(model_name="VerificadoProfesional/SaBERT-Spanish-Sentiment-Analysis")
    
    train_dataset, val_dataset = classifier.load_and_preprocess_data(
        data_path="data/train.csv",
        text_column="observaciones",  # Column with complaint text
        label_column="areaServicioDescripcion",  # Column with complaint category
        test_size=0.2
    )
    
    # Apply domain adaptation techniques
    enhanced_train_dataset = classifier.apply_domain_adaptation(train_dataset)
    
    # Train the model
    classifier.train(
        train_dataset=enhanced_train_dataset,
        val_dataset=val_dataset,
        batch_size=16,
        epochs=3
    )
    
    # Save the trained model
    classifier.save_model("model")
    
    # Make predictions on social media text examples
    social_media_examples = [
        'Hace un año y medio tengo hecho un reclamo x vereda y pared de mi casa rotas x los árboles. Quien se va a hacer cargo si alguien se tropieza? Saquen los árboles y arreglen la vereda. Exp 15397/2023. Incidente número 7275781',
        'Hace 7 meses estoy reclamando por un Ficus (especie ilegal para estar en la vía pública) que me está rompiendo todos los caños de mi casa y nadie hace nada.',
        'Una basura el barrio, calles rotas, basura por todos lados, inseguridad cada vez más violenta, asco SAN ISIDRO ES UNA MIERDA',
        'Basura por todos lados',
        'Está gestión no es mejor!! Venga a recorrer Boulogne la mugre , veredas todas rotas.',
        'Por favor arreglar la mitad de la calle Uruguay que le pertenece al partido de San Isidro y está llena de pozos'
    ]
    
    results = classifier.predict(social_media_examples)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Predicted category: {result['predicted_category']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("---")

if __name__ == "__main__":
    main()