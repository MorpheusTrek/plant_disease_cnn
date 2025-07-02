import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    f1_score, precision_score, recall_score,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import random

def plot_training_history(history, name):
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)  # Set minimum y-axis to 0
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)  # Set minimum y-axis to 0
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("plots", f"{name}_l_and_a.png"), dpi=300, bbox_inches='tight')
    plt.show()


def show_sample_predictions(dataset, model, class_names, num_samples=5):
    # Collect a batch of images and labels
    all_images = []
    all_labels = []
    
    for images, labels in dataset:
        all_images.extend(images)
        all_labels.extend(labels)
    
    # Randomly select indices
    indices = random.sample(range(len(all_images)), num_samples)

    plt.figure(figsize=(15, 5))
    predictions = model.predict(tf.stack([all_images[i] for i in indices]))

    for i, idx in enumerate(indices):
        image = all_images[idx]
        label = all_labels[idx]

        # Determine label format (one-hot or scalar)
        true_label_index = label.numpy() if label.shape == () else np.argmax(label)
        pred_label_index = np.argmax(predictions[i])

        true_label = class_names[true_label_index]
        pred_label = class_names[pred_label_index]

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        plt.axis("off")

    plt.show()


def comprehensive_evaluation(model, test_dataset, class_names, name, save_plots=True):
    """
    Comprehensive evaluation of a classification model
    """
    print("🔍 Starting Comprehensive Model Evaluation...")
    print("=" * 60)
    
    # Get predictions and true labels
    y_true = []
    y_pred_proba = []
    
    print("📊 Collecting predictions...")
    for batch_images, batch_labels in test_dataset:
        predictions = model.predict(batch_images, verbose=0)
        y_pred_proba.extend(predictions)
        y_true.extend(batch_labels.numpy())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    num_classes = len(class_names)
    
    # 1. BASIC METRICS
    print("\n📈 BASIC CLASSIFICATION METRICS")
    print("-" * 40)
    
    accuracy = np.mean(y_true == y_pred)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Macro and weighted averages
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    
    print(f"F1-Score (Macro):     {f1_macro:.4f}")
    print(f"F1-Score (Weighted):  {f1_weighted:.4f}")
    print(f"Precision (Macro):    {precision_macro:.4f}")
    print(f"Recall (Macro):       {recall_macro:.4f}")
    
    # 2. DETAILED CLASSIFICATION REPORT
    print("\n📋 DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # 3. CONFUSION MATRIX
    print("\n🔢 CONFUSION MATRIX")
    print("-" * 40)
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by true labels (row-wise percentages)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert to %

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f',  # 1 decimal place for percentages
        cmap='Blues', 
        vmin=0, 
        vmax=100,  # Fix scale to 0-100%
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}
    )
    plt.title('Confusion Matrix (% of True Labels)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join("plots", f'{name}_conf_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print confusion matrix details
    print("Confusion Matrix (Raw Counts):")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {cm[i]}")
    
    # 4. PER-CLASS METRICS
    print("\n📊 PER-CLASS DETAILED METRICS")
    print("-" * 40)
    
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        class_predictions = (y_pred == i)
        
        # True/False Positives/Negatives
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        tn = np.sum((y_true != i) & (y_pred != i))
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n{class_name.upper()}:")
        print(f"  Samples: {np.sum(class_mask)}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (Sensitivity): {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  True Positives: {tp}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
    
    # 5. ROC CURVES (for multi-class)
    if num_classes > 2:
        print("\n📈 ROC ANALYSIS (Multi-class)")
        print("-" * 40)
        
        # Binarize labels for multi-class ROC
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, 
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join("plots", f'{name}_roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Macro-average ROC AUC
        macro_roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
        print(f"Macro-average ROC AUC: {macro_roc_auc:.4f}")
    
    # 6. PRECISION-RECALL CURVES
    print("\n📊 PRECISION-RECALL ANALYSIS")
    print("-" * 40)
    
    if num_classes > 2:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        
        plt.figure(figsize=(12, 8))
        
        for i in range(num_classes):
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true_bin[:, i], y_pred_proba[:, i]
            )
            avg_precision = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall_curve, precision_curve, linewidth=2,
                    label=f'{class_names[i]} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join("plots", f'{name}_prec_recall.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # 7. MODEL CONFIDENCE ANALYSIS
    print("\n🎯 MODEL CONFIDENCE ANALYSIS")
    print("-" * 40)
    
    max_probs = np.max(y_pred_proba, axis=1)
    correct_predictions = (y_true == y_pred)
    
    print(f"Average confidence (all predictions): {np.mean(max_probs):.4f}")
    print(f"Average confidence (correct predictions): {np.mean(max_probs[correct_predictions]):.4f}")
    print(f"Average confidence (incorrect predictions): {np.mean(max_probs[~correct_predictions]):.4f}")
    
    # Confidence distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(max_probs[correct_predictions], bins=20, alpha=0.7, label='Correct', color='green')
    plt.hist(max_probs[~correct_predictions], bins=20, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Model Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracy = []
    bin_counts = []
    
    for i in range(len(confidence_bins)-1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
        if np.sum(mask) > 0:
            bin_acc = np.mean(correct_predictions[mask])
            bin_accuracy.append(bin_acc)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracy.append(0)
            bin_counts.append(0)
    
    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    plt.bar(bin_centers, bin_accuracy, width=0.08, alpha=0.7, color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence Bin')
    plt.ylabel('Accuracy')
    plt.title('Calibration Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join("plots", f'{name}_confidence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 8. SUMMARY STATISTICS
    print("\n📋 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"✅ Total Test Samples: {len(y_true)}")
    print(f"✅ Number of Classes: {num_classes}")
    print(f"✅ Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✅ Macro F1-Score: {f1_macro:.4f}")
    print(f"✅ Weighted F1-Score: {f1_weighted:.4f}")
    if num_classes > 2:
        print(f"✅ Macro ROC-AUC: {macro_roc_auc:.4f}")
    print(f"✅ Average Confidence: {np.mean(max_probs):.4f}")
    
    # Return key metrics
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
    }

    

    with open(f'results\\{name}.json', 'w') as f:
        json.dump(results, f)
    
    return results