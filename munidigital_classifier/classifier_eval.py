import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve, 
    auc, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    confusion_matrix
)
import argparse
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import from your existing model
from classifier import ComplaintClassifier, ComplaintDataset

def load_model_and_tokenizer(model_dir):
    """Load the trained model and tokenizer"""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Load label mappings
    with open(f"{model_dir}/label_mappings.json", "r") as f:
        mappings = json.load(f)
        label_encoder = mappings["label_encoder"]
        label_decoder = mappings["label_decoder"]
    
    # Convert string keys back to integers for label_decoder if needed
    if all(k.isdigit() for k in label_decoder.keys()):
        label_decoder = {int(k): v for k, v in label_decoder.items()}
    
    # Use the ComplaintClassifier class to load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ComplaintClassifier(model_name=model_dir, num_labels=len(label_encoder))
    classifier.load_model(model_dir)
    
    print(f"Model loaded from {model_dir} to {device}")
    return classifier

def load_test_data(data_path, text_column, label_column, classifier):
    """Load and preprocess test data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure text column exists
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in data file")
    
    # Ensure label column exists
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in data file")
    
    # Clean data
    df[text_column] = df[text_column].fillna("")
    df[text_column] = df[text_column].apply(classifier._clean_text)
    
    # Convert labels to numerical values
    df['label_id'] = df[label_column].map(classifier.label_encoder)
    
    # Create dataset
    test_dataset = ComplaintDataset(
        texts=df[text_column].tolist(),
        labels=df['label_id'].tolist(),
        tokenizer=classifier.tokenizer,
        max_length=classifier.max_length
    )
    
    print(f"Test data loaded with {len(df)} samples")
    
    return test_dataset, df

def evaluate_model(classifier, test_dataset, batch_size=16):
    """Evaluate model performance with multiple metrics"""
    device = classifier.device
    model = classifier.model
    tokenizer = classifier.tokenizer
    label_decoder = classifier.label_decoder
    
    # Set model to evaluation mode
    model.eval()
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Collect all predictions and true labels
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Process batches
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Get model outputs
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Get predicted classes
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            # Store results
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to arrays for easier processing
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    label_decoder = {int(k): v for k, v in label_decoder.items()}

    # Get human-readable labels
    y_true_labels = [label_decoder[int(i)] for i in y_true]
    y_pred_labels = [label_decoder[int(i)] for i in y_pred]
    
    # Calculate metrics for each class
    class_metrics = {}
    num_classes = len(label_decoder)
    
    # Binary classification metrics for each class (one-vs-rest)
    for class_idx in range(num_classes):
        class_name = label_decoder[class_idx]
        
        # Create binary labels for this class
        binary_true = (y_true == class_idx).astype(int)
        binary_pred = (y_pred == class_idx).astype(int)
        
        # Class-specific probabilities
        class_probs = y_probs[:, class_idx]
        
        # Calculate binary metrics
        precision = precision_score(binary_true, binary_pred, zero_division=0)
        recall = recall_score(binary_true, binary_pred, zero_division=0)
        f1 = f1_score(binary_true, binary_pred, zero_division=0)
        
        # Calculate PR and ROC curves
        precisions, recalls, pr_thresholds = precision_recall_curve(binary_true, class_probs)
        try:
            pr_auc = auc(recalls, precisions)
        except:
            pr_auc = 0.0
            
        fpr, tpr, roc_thresholds = roc_curve(binary_true, class_probs)
        try:
            roc_auc = auc(fpr, tpr)
        except:
            roc_auc = 0.0
        
        # Store metrics
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'pr_curve': (precisions, recalls),
            'roc_curve': (fpr, tpr)
        }
    
    # Print overall classification report
    print("\nClassification Report:")
    print(classification_report(y_true_labels, y_pred_labels))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Calculate weighted average metrics
    class_counts = np.bincount(y_true, minlength=num_classes)
    weights = class_counts / len(y_true)
    
    weighted_precision = sum(class_metrics[label_decoder[i]]['precision'] * weights[i] for i in range(num_classes))
    weighted_recall = sum(class_metrics[label_decoder[i]]['recall'] * weights[i] for i in range(num_classes))
    weighted_f1 = sum(class_metrics[label_decoder[i]]['f1'] * weights[i] for i in range(num_classes))
    weighted_pr_auc = sum(class_metrics[label_decoder[i]]['pr_auc'] * weights[i] for i in range(num_classes))
    weighted_roc_auc = sum(class_metrics[label_decoder[i]]['roc_auc'] * weights[i] for i in range(num_classes))
    
    # Print weighted metrics
    print("\nWeighted Average Metrics:")
    print(f"Precision: {weighted_precision:.4f}")
    print(f"Recall: {weighted_recall:.4f}")
    print(f"F1 Score: {weighted_f1:.4f}")
    print(f"PR AUC: {weighted_pr_auc:.4f}")
    print(f"ROC AUC: {weighted_roc_auc:.4f}")
    
    # Plot PR and ROC curves for each class
    plt.figure(figsize=(20, 10))
    
    # PR curves
    plt.subplot(1, 2, 1)
    for class_idx in range(min(num_classes, 5)):  # Limit to 5 classes for readability
        class_name = label_decoder[class_idx]
        precisions, recalls = class_metrics[class_name]['pr_curve']
        pr_auc = class_metrics[class_name]['pr_auc']
        plt.plot(recalls, precisions, label=f"{class_name} (AUC = {pr_auc:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (top 5 classes)')
    plt.grid(True)
    plt.legend(loc='lower left')
    
    # ROC curves
    plt.subplot(1, 2, 2)
    for class_idx in range(min(num_classes, 5)):  # Limit to 5 classes for readability
        class_name = label_decoder[class_idx]
        fpr, tpr = class_metrics[class_name]['roc_curve']
        roc_auc = class_metrics[class_name]['roc_auc']
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (top 5 classes)')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = 'transformer_model_performance.png'
    plt.savefig(plot_file)
    print(f"\nPerformance curves saved to {plot_file}")
    
    # Return metrics for further analysis
    return {
        'class_metrics': class_metrics,
        'weighted_metrics': {
            'precision': weighted_precision,
            'recall': weighted_recall,
            'f1': weighted_f1,
            'pr_auc': weighted_pr_auc,
            'roc_auc': weighted_roc_auc
        },
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }

def plot_class_distribution(y_true, y_pred, label_decoder):
    """Plot the distribution of classes in true vs predicted labels"""
    num_classes = len(label_decoder)
    
    # Count occurrences of each class
    true_counts = np.bincount(y_true, minlength=num_classes)
    pred_counts = np.bincount(y_pred, minlength=num_classes)
    
    label_decoder = {int(k): v for k, v in label_decoder.items()}
    # Get class names
    class_names = [label_decoder[int(i)] for i in range(num_classes)]
    
    # Sort classes by true count for better visualization
    sorted_idx = np.argsort(-true_counts)
    top_classes = sorted_idx[:10]  # Show top 10 classes
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(top_classes))
    width = 0.35
    
    # Plot bar chart
    plt.bar(x - width/2, true_counts[top_classes], width, label='True')
    plt.bar(x + width/2, pred_counts[top_classes], width, label='Predicted')
    
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution: True vs Predicted')
    plt.xticks(x, [class_names[i] for i in top_classes], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    dist_file = 'class_distribution.png'
    plt.savefig(dist_file)
    print(f"Class distribution plot saved to {dist_file}")

def analyze_misclassifications(test_df, y_true, y_pred, y_probs, label_decoder, top_n=5):
    """Analyze the most severe misclassifications"""
    num_classes = len(label_decoder)
    
    # Create a DataFrame with predictions
    results_df = test_df.copy()
    results_df['true_label_id'] = y_true
    results_df['pred_label_id'] = y_pred
    results_df['confidence'] = np.max(y_probs, axis=1)
    
    # Add human-readable labels
    results_df['true_label'] = results_df['true_label_id'].map(label_decoder)
    results_df['pred_label'] = results_df['pred_label_id'].map(label_decoder)
    
    # Identify misclassifications
    results_df['is_correct'] = (results_df['true_label_id'] == results_df['pred_label_id'])
    misclassified = results_df[~results_df['is_correct']]
    
    print(f"\nMisclassification Analysis:")
    print(f"Total misclassifications: {len(misclassified)} out of {len(results_df)} ({len(misclassified)/len(results_df)*100:.2f}%)")
    
    # Find most common error types (true -> predicted)
    error_types = misclassified.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
    error_types = error_types.sort_values('count', ascending=False)
    
    print("\nTop 10 Most Common Error Types:")
    for i, row in error_types.head(10).iterrows():
        print(f"{row['true_label']} -> {row['pred_label']}: {row['count']} instances")
    
    # Find high-confidence errors
    high_conf_errors = misclassified.sort_values('confidence', ascending=False)
    
    print(f"\nTop {top_n} Highest Confidence Errors:")
    for i, row in high_conf_errors.head(top_n).iterrows():
        print(f"\nConfidence: {row['confidence']:.3f}")
        print(f"True: {row['true_label']} | Predicted: {row['pred_label']}")
        print(f"Text: {row['text'][:100]}...")
    
    return misclassified, error_types

def save_detailed_results(test_df, y_true, y_pred, y_probs, label_decoder, output_file='detailed_results.csv'):
    """Save detailed prediction results to CSV file"""
    # Create a DataFrame with all predictions
    results_df = test_df.copy()
    results_df['true_label_id'] = y_true
    results_df['pred_label_id'] = y_pred
    results_df['confidence'] = np.max(y_probs, axis=1)
    
    label_decoder = {int(k): v for k, v in label_decoder.items()}
    # Add human-readable labels
    results_df['true_label'] = results_df['true_label_id'].map(label_decoder)
    results_df['pred_label'] = results_df['pred_label_id'].map(label_decoder)
    
    # Add correctness flag
    results_df['is_correct'] = (results_df['true_label_id'] == results_df['pred_label_id'])
    
    # Add probabilities for each class
    for i in range(y_probs.shape[1]):
        class_name = label_decoder[i]
        results_df[f'prob_{class_name}'] = y_probs[:, i]
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to {output_file}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer-based Complaint Classifier')
    parser.add_argument('--model', help='Path to the trained model directory', 
                        default='model')
    parser.add_argument('--data', help='Path to test data CSV file', required=True)
    parser.add_argument('--text-col', help='Name of column containing text data',
                        default='observaciones')
    parser.add_argument('--label-col', help='Name of column containing label data',
                        default='areaServicioDescripcion')
    parser.add_argument('--batch-size', help='Batch size for evaluation',
                        type=int, default=16)
    parser.add_argument('--analyze-errors', help='Analyze misclassifications',
                        action='store_true')
    parser.add_argument('--output', help='Output file for detailed results',
                        default='transformer_detailed_results.csv')
    
    args = parser.parse_args()
    
    # Load model
    classifier = load_model_and_tokenizer(args.model)
    
    # Load test data
    test_dataset, test_df = load_test_data(
        args.data, 
        args.text_col, 
        args.label_col, 
        classifier
    )
    
    # Evaluate model
    results = evaluate_model(classifier, test_dataset, batch_size=args.batch_size)
    
    # Plot class distribution
    plot_class_distribution(
        results['y_true'], 
        results['y_pred'], 
        classifier.label_decoder
    )
    
    # Analyze errors if requested
    if args.analyze_errors:
        analyze_misclassifications(
            test_df,
            results['y_true'],
            results['y_pred'],
            results['y_probs'],
            classifier.label_decoder
        )
    
    # Save detailed results
    save_detailed_results(
        test_df,
        results['y_true'],
        results['y_pred'],
        results['y_probs'],
        classifier.label_decoder,
        args.output
    )
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()