import sys
import torch
from datasets import load_dataset, Dataset
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
)
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# --- Configuration Variables ---
HF_TOKEN = os.getenv("HF_TOKEN")

# Path to your saved checkpoint
checkpoint_path = "checkpoints"  # Or "checkpoints/checkpoint-68671" for specific checkpoint

# Model configuration
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
load_in_8bit = False
load_in_4bit = True

# Label mapping
id_to_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
label_to_id = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

# Question template (same as training)
question_template = (
    "### Human: Classify the relationship between the following two sentences "
    "as one of the following: entailment, neutral, contradiction. "
)


def format_instruction(premise, hypothesis):
    """Formats the data into an instruction-following prompt (without label)."""
    text = (
        f'{question_template}\n'
        f'premise: {premise}\n'
        f'hypothesis: {hypothesis}\n\n'
        f'### Assistant:'
    )
    return text


def load_model_and_tokenizer(checkpoint_path, base_model_name):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading tokenizer from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        token=HF_TOKEN,
        padding_side='right',
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model {base_model_name}...")
    
    # Quantization config
    quantization_config = None
    device_map = None
    torch_dtype = None
    
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
        device_map = {"": 0}
        torch_dtype = torch.bfloat16
        print("Using BitsAndBytes quantization.")
    else:
        torch_dtype = torch.bfloat16
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch_dtype,
        token=HF_TOKEN
    )
    
    # Load LoRA adapter
    print(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer


def extract_prediction(generated_text):
    """Extract the predicted label from generated text."""
    try:
        # Get text after '### Assistant:'
        result = generated_text.split('### Assistant:')[-1].strip().lower()
        
        # Check which label is in the result
        for label in ['entailment', 'neutral', 'contradiction']:
            if label in result:
                return label
        
        # If no exact match, return the result as is
        return result
    except Exception as e:
        print(f"Error extracting prediction: {e}")
        return ""


def evaluate_model(model, tokenizer, test_dataset, batch_size=8):
    """Evaluate the model on test dataset."""
    model.eval()
    
    print("\n--- Starting Test Evaluation ---")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map={'': 0},
    )
    
    # Prepare queries
    print("Preparing test queries...")
    queries = []
    true_labels = []
    
    for example in tqdm(test_dataset, desc="Preparing data"):
        if example['label'] != -1:  # Skip undefined labels
            query = format_instruction(example['premise'], example['hypothesis'])
            queries.append(query)
            true_labels.append(id_to_label[example['label']])
    
    print(f"Number of valid test examples: {len(queries)}")
    print(f"\nExample query:\n{queries[0]}\n")
    
    # Generate predictions in batches
    print("Generating predictions...")
    predictions = []
    
    for i in tqdm(range(0, len(queries), batch_size), desc="Inference"):
        batch_queries = queries[i:i + batch_size]
        
        sequences = pipe(
            batch_queries,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=5,  # Enough for label generation
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            temperature=None,  # Explicitly disable since we're not sampling
            top_p=None,  # Explicitly disable since we're not sampling
        )
        
        # Extract predictions from batch
        for seq in sequences:
            generated_text = seq[0]['generated_text']
            pred = extract_prediction(generated_text)
            predictions.append(pred)
    
    return predictions, true_labels


def calculate_metrics(predictions, true_labels):
    """Calculate and display evaluation metrics."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    
    # Convert to numerical labels for sklearn
    label_list = ['entailment', 'neutral', 'contradiction']
    y_true = [label_to_id.get(label, -1) for label in true_labels]
    y_pred = [label_to_id.get(pred, -1) for pred in predictions]
    
    # Calculate accuracy
    correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
    accuracy = correct / len(true_labels)
    
    print(f"\nTotal Examples: {len(true_labels)}")
    print(f"Correct Predictions: {correct}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT:")
    print("-" * 60)
    print(classification_report(
        y_true, 
        y_pred, 
        target_names=label_list,
        digits=4
    ))
    
    # Confusion matrix
    print("-" * 60)
    print("CONFUSION MATRIX:")
    print("-" * 60)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Print confusion matrix with labels
    print(f"\n{'':12} {'Predicted':>40}")
    print(f"{'True':12} {'entailment':>12} {'neutral':>12} {'contradiction':>12}")
    print("-" * 60)
    for i, true_label in enumerate(label_list):
        print(f"{true_label:12}", end="")
        for j in range(3):
            print(f"{cm[i][j]:>12}", end="")
        print()
    
    # Show some example predictions
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS:")
    print("-" * 60)
    
    # Show first 10 predictions
    for i in range(min(10, len(predictions))):
        status = "✓" if predictions[i] == true_labels[i] else "✗"
        print(f"{status} True: {true_labels[i]:15} | Predicted: {predictions[i]:15}")
    
    # Show some errors if any
    errors = [(i, t, p) for i, (t, p) in enumerate(zip(true_labels, predictions)) if t != p]
    if errors:
        print(f"\n{len(errors)} errors found. Showing first 5:")
        for idx, true_label, pred in errors[:5]:
            print(f"  Index {idx}: True={true_label}, Predicted={pred}")
    
    print("\n" + "=" * 60)
    
    return accuracy


def main():
    """Main function to run model testing."""
    print("=" * 60)
    print("SNLI MODEL TESTING")
    print("=" * 60)
    
    # Load test dataset
    print("\nLoading SNLI test dataset...")
    snli = load_dataset("stanfordnlp/snli")
    test_data = snli["test"]
    print(f"Test dataset loaded: {len(test_data)} examples")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, base_model_name)
    
    # Evaluate on test set
    predictions, true_labels = evaluate_model(model, tokenizer, test_data)
    
    # Calculate and display metrics
    accuracy = calculate_metrics(predictions, true_labels)
    
    # Save results to file
    results_file = "test_results.txt"
    print(f"\nSaving results to {results_file}...")
    with open(results_file, 'w') as f:
        f.write(f"SNLI Test Set Results\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Total Examples: {len(true_labels)}\n")
        f.write(f"Correct Predictions: {sum(1 for t, p in zip(true_labels, predictions) if t == p)}\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"\nDetailed predictions saved separately.\n")
    
    # Save detailed predictions
    detailed_file = "detailed_predictions.txt"
    print(f"Saving detailed predictions to {detailed_file}...")
    with open(detailed_file, 'w') as f:
        for i, (true_label, pred) in enumerate(zip(true_labels, predictions)):
            status = "CORRECT" if true_label == pred else "INCORRECT"
            f.write(f"{i}\t{true_label}\t{pred}\t{status}\n")
    
    print("\n✓ Testing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(int(main() or 0))
