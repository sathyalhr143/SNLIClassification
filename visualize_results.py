import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_predictions(filename='detailed_predictions.txt'):
    """Load predictions from file."""
    predictions = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                predictions.append({
                    'index': int(parts[0]),
                    'true': parts[1],
                    'predicted': parts[2],
                    'status': parts[3]
                })
    return pd.DataFrame(predictions)


def plot_confusion_matrix(df, save_path='confusion_matrix.png'):
    """Create and save confusion matrix heatmap."""
    labels = ['entailment', 'neutral', 'contradiction']
    label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    y_true = [label_to_id[label] for label in df['true']]
    y_pred = [label_to_id[label] for label in df['predicted']]
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - SNLI Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()


def plot_class_performance(df, save_path='class_performance.png'):
    """Plot per-class accuracy."""
    labels = ['entailment', 'neutral', 'contradiction']
    accuracies = []
    
    for label in labels:
        subset = df[df['true'] == label]
        accuracy = len(subset[subset['status'] == 'CORRECT']) / len(subset)
        accuracies.append(accuracy * 100)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, accuracies, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Class performance plot saved to {save_path}")
    plt.close()


def plot_error_distribution(df, save_path='error_distribution.png'):
    """Plot distribution of error types."""
    errors = df[df['status'] == 'INCORRECT']
    error_types = errors.groupby(['true', 'predicted']).size().sort_values(ascending=False)
    
    # Create labels for error types
    labels = [f"{true} → {pred}" for true, pred in error_types.index]
    counts = error_types.values
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(labels, counts, color='#e74c3c', alpha=0.7)
    plt.xlabel('Number of Errors', fontsize=12)
    plt.ylabel('Error Type', fontsize=12)
    plt.title('Distribution of Error Types', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f' {count}', ha='left', va='center', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Error distribution plot saved to {save_path}")
    plt.close()


def plot_overall_metrics(df, save_path='overall_metrics.png'):
    """Create a summary metrics visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall accuracy
    correct = len(df[df['status'] == 'CORRECT'])
    incorrect = len(df[df['status'] == 'INCORRECT'])
    accuracy = correct / len(df) * 100
    
    # Pie chart
    ax1 = axes[0]
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax1.pie([correct, incorrect], 
                                         labels=['Correct', 'Incorrect'],
                                         colors=colors, autopct='%1.1f%%',
                                         startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax1.set_title(f'Overall Accuracy: {accuracy:.2f}%', fontsize=14, fontweight='bold')
    
    # Bar chart of samples per class
    ax2 = axes[1]
    labels = ['entailment', 'neutral', 'contradiction']
    class_counts = [len(df[df['true'] == label]) for label in labels]
    colors_bar = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax2.bar(labels, class_counts, color=colors_bar, alpha=0.8)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_xlabel('Class', fontsize=12)
    ax2.set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Overall metrics plot saved to {save_path}")
    plt.close()


def create_summary_report(df):
    """Create a comprehensive summary report."""
    print("\n" + "=" * 80)
    print("VISUALIZATION SUMMARY")
    print("=" * 80)
    
    total = len(df)
    correct = len(df[df['status'] == 'CORRECT'])
    accuracy = correct / total * 100
    
    print(f"\nTotal Examples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    print("\nPer-Class Breakdown:")
    labels = ['entailment', 'neutral', 'contradiction']
    for label in labels:
        subset = df[df['true'] == label]
        class_correct = len(subset[subset['status'] == 'CORRECT'])
        class_acc = class_correct / len(subset) * 100
        print(f"  {label:15}: {class_correct:4}/{len(subset):4} ({class_acc:6.2f}%)")
    
    print("\n" + "=" * 80)


def main():
    """Main function to create all visualizations."""
    print("=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Load predictions
    print("\nLoading predictions...")
    df = load_predictions()
    
    # Create visualizations
    print("\nGenerating plots...")
    plot_confusion_matrix(df)
    plot_class_performance(df)
    plot_error_distribution(df)
    plot_overall_metrics(df)
    
    # Print summary
    create_summary_report(df)
    
    print("\n✓ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - class_performance.png")
    print("  - error_distribution.png")
    print("  - overall_metrics.png")
    
    return 0


if __name__ == "__main__":
    sys.exit(int(main() or 0))
