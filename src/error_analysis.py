import sys
import os
from datasets import load_dataset
from collections import defaultdict
import pandas as pd

# Output folder for results
output_folder = "test_results_epoch_3"
os.makedirs(output_folder, exist_ok=True)

def load_predictions(filename='test_results_epoch_3/detailed_predictions.txt'):
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


def analyze_errors(df, test_data):
    """Analyze error patterns."""
    errors = df[df['status'] == 'INCORRECT'].copy()
    
    print("=" * 80)
    print("ERROR ANALYSIS REPORT")
    print("=" * 80)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"Correct predictions: {len(df[df['status'] == 'CORRECT'])}")
    print(f"Incorrect predictions: {len(errors)}")
    print(f"Accuracy: {len(df[df['status'] == 'CORRECT']) / len(df) * 100:.2f}%")
    
    # Error breakdown by type
    print("\n" + "-" * 80)
    print("ERROR BREAKDOWN BY TYPE:")
    print("-" * 80)
    error_types = errors.groupby(['true', 'predicted']).size().sort_values(ascending=False)
    for (true_label, pred_label), count in error_types.items():
        percentage = count / len(errors) * 100
        print(f"{true_label:15} → {pred_label:15}: {count:4} errors ({percentage:5.2f}%)")
    
    # Analyze text length correlation
    print("\n" + "-" * 80)
    print("TEXT LENGTH ANALYSIS:")
    print("-" * 80)
    
    error_premise_lengths = []
    error_hypothesis_lengths = []
    correct_premise_lengths = []
    correct_hypothesis_lengths = []
    
    for idx, row in df.iterrows():
        example = test_data[row['index']]
        premise_len = len(example['premise'].split())
        hypothesis_len = len(example['hypothesis'].split())
        
        if row['status'] == 'INCORRECT':
            error_premise_lengths.append(premise_len)
            error_hypothesis_lengths.append(hypothesis_len)
        else:
            correct_premise_lengths.append(premise_len)
            correct_hypothesis_lengths.append(hypothesis_len)
    
    print(f"Average premise length (errors):   {sum(error_premise_lengths)/len(error_premise_lengths):.2f} words")
    print(f"Average premise length (correct):  {sum(correct_premise_lengths)/len(correct_premise_lengths):.2f} words")
    print(f"Average hypothesis length (errors):   {sum(error_hypothesis_lengths)/len(error_hypothesis_lengths):.2f} words")
    print(f"Average hypothesis length (correct):  {sum(correct_hypothesis_lengths)/len(correct_hypothesis_lengths):.2f} words")
    
    return errors


def show_sample_errors(errors, test_data, num_samples=20):
    """Display sample errors with their text."""
    print("\n" + "=" * 80)
    print(f"SAMPLE ERRORS (showing {min(num_samples, len(errors))} of {len(errors)})")
    print("=" * 80)
    
    for idx, (_, row) in enumerate(errors.head(num_samples).iterrows()):
        example = test_data[row['index']]
        print(f"\n[Error {idx+1}] Index: {row['index']}")
        print(f"True Label: {row['true'].upper()} | Predicted: {row['predicted'].upper()}")
        print(f"Premise: {example['premise']}")
        print(f"Hypothesis: {example['hypothesis']}")
        print("-" * 80)


def analyze_specific_confusion(errors, test_data, true_label, predicted_label, num_samples=10):
    """Analyze specific confusion pair."""
    specific_errors = errors[(errors['true'] == true_label) & 
                             (errors['predicted'] == predicted_label)]
    
    if len(specific_errors) == 0:
        print(f"\nNo errors found for {true_label} → {predicted_label}")
        return
    
    print("\n" + "=" * 80)
    print(f"ANALYZING: {true_label.upper()} misclassified as {predicted_label.upper()}")
    print(f"Total occurrences: {len(specific_errors)}")
    print("=" * 80)
    
    for idx, (_, row) in enumerate(specific_errors.head(num_samples).iterrows()):
        example = test_data[row['index']]
        print(f"\n[Example {idx+1}]")
        print(f"Premise: {example['premise']}")
        print(f"Hypothesis: {example['hypothesis']}")
        print("-" * 80)


def main():
    print("Loading predictions and test data...")
    
    # Load predictions
    df = load_predictions()
    
    # Load test dataset
    snli = load_dataset("stanfordnlp/snli")
    test_data = snli["test"]
    
    # Run analysis
    errors = analyze_errors(df, test_data)
    
    # Show sample errors
    show_sample_errors(errors, test_data, num_samples=15)
    
    # Analyze specific confusions (most common error types)
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF MOST COMMON ERRORS")
    print("=" * 80)
    
    # Get top 3 error types
    error_types = errors.groupby(['true', 'predicted']).size().sort_values(ascending=False).head(3)
    
    for (true_label, pred_label), count in error_types.items():
        analyze_specific_confusion(errors, test_data, true_label, pred_label, num_samples=5)
    
    # Save detailed error report
    error_report_file = os.path.join(output_folder, "error_analysis_report.txt")
    print(f"\n\nSaving detailed error report to {error_report_file}...")
    
    with open(error_report_file, 'w') as f:
        f.write("DETAILED ERROR ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, (_, row) in enumerate(errors.iterrows()):
            example = test_data[row['index']]
            f.write(f"Error {idx+1} (Index: {row['index']})\n")
            f.write(f"True: {row['true']} | Predicted: {row['predicted']}\n")
            f.write(f"Premise: {example['premise']}\n")
            f.write(f"Hypothesis: {example['hypothesis']}\n")
            f.write("-" * 80 + "\n\n")
    
    print(f"✓ Error analysis complete! Report saved to {error_report_file}")
    return 0


if __name__ == "__main__":
    sys.exit(int(main() or 0))
