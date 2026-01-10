import sys
import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
    )

from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()


# --- Configuration Variables ---
# NOTE: Replace with a valid Hugging Face token
# The token is passed to login() and the model/tokenizer loading functions.

HF_TOKEN = os.getenv("HF_TOKEN") # **Replace with your actualtoken**

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# Training Hyperparameters
dataset_text_field = "text"
log_with = 'wandb'
learning_rate = 1.41e-5
batch_size = 4
seq_length = 512
gradient_accumulation_steps = 2
num_train_epochs = 1 ###################need to change to 3 for better results
logging_steps = 100
output_dir = "checkpoints"

report_to = 'wandb'  # Options: 'wandb', 'tensorboard', 'all', or 'none'


# Quantization
load_in_8bit = False
load_in_4bit = True


# PEFT/LoRA Config
use_peft = True
peft_lora_r = 64
peft_lora_alpha = 16


# --- Main Function ---
def main():
    tqdm.pandas()

    # Log in to Hugging Face Hub
    login(token=HF_TOKEN)

    ## Data Preparation ##
    print("Loading dataset...")
    snli = load_dataset("stanfordnlp/snli")
    train_data = snli["train"]
    validation_data = snli["validation"]
    id_to_label = {0:'entailment', 1:'neutral', 2:'contradiction'}


    # Use the official chat template format for Instruct models
    question_template = (
        "### Human: Classify the relationship between the following two sentences "
        "as one of the following: entailment, neutral, contradiction. "
        )

    def format_instruction(premise, hypothesis, label):
        """Formats the data into an instruction-following prompt."""
        if label == -1: # Skip examples with undefined label
            return None
        
        # Using a simple structure for demonstration, consider using the official
        # model chat template if possible (see below for an alternative using tokenizer)
        text = (
            f'{question_template}\n'
            f'premise: {premise}\n'
            f'hypothesis: {hypothesis}\n\n'
            f'### Assistant: {id_to_label[label]}'
            )
        return text 
    

    # Filter out -1 labels and create the instruction dataset
    train_instructions = [
        format_instruction(x, y, z)
        for x, y, z in zip(train_data['premise'], train_data['hypothesis'],
        train_data['label'])
        if z != -1
        ]
    
    validation_instructions = [
        format_instruction(x, y, z)
        for x, y, z in zip(validation_data['premise'],
        validation_data['hypothesis'], validation_data['label'])
        if z != -1
        ]
    
    ds_train = Dataset.from_dict({"text": train_instructions})
    ds_validation = Dataset.from_dict({"text": validation_instructions})
    instructions_ds_dict = DatasetDict({"train": ds_train, "eval": ds_validation})
    print(f"Example training instruction:\n{instructions_ds_dict['train']['text'][0]}\n")


    ## Model and Tokenizer Setup ##
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        # Set padding_side to 'right' for Causal LM training (important for models like Llama)
        padding_side='right',
        )
    
    

    # Recommended to set a new pad token if not already defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine Quantization Config
    if load_in_8bit and load_in_4bit:
        raise ValueError("Select either 8 bits or 4 bits for quantization, not both.")
    
    quantization_config = None
    device_map = None
    torch_dtype = None

    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
            )
    
        device_map = {"": 0}
        # Use bfloat16 for modern GPUs/quantization compatibility
        torch_dtype = torch.bfloat16
        print("Using BitsAndBytes quantization.")
    else:
    # Use recommended bfloat16 for better numerical stability if no quantization
        torch_dtype = torch.bfloat16    
    print("Loading model...")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch_dtype, # <-- Changed to 'dtype'
        token=HF_TOKEN
        )
    
    # The Llama-3.1 model has an input size of 8192, we can set it to the max sequence length
    model.config.use_cache = False # Recommended to disable cache for training
    ## PEFT Configuration ##
    peft_config = None
    if use_peft:
        peft_config = LoraConfig(
            r=peft_lora_r,
            lora_alpha=peft_lora_alpha,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
            )
        print("Using PEFT (LoRA) configuration.")

        ## SFTTrainer Setup (Using SFTConfig) ##
    print("Setting up SFTTrainer...")

    # Pass all training/SFT-specific arguments via SFTConfig
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        num_train_epochs=num_train_epochs,
        max_length=seq_length,
        dataset_text_field=dataset_text_field,
        report_to=report_to,
        # Checkpoint saving configuration
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,  # Only keep the most recent checkpoint
        # Other TrainingArguments parameters can be set here:
        # e.g., evaluation_strategy="steps"
        )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=instructions_ds_dict['train'],
        eval_dataset=instructions_ds_dict['eval'],
        args=sft_config, # Pass SFTConfig here
        peft_config=peft_config,
        )
    
    ## Training and Evaluation ##
    print("Starting training...")
    
    # Check for existing checkpoints and resume if available
    checkpoint_dir = None
    if os.path.exists(output_dir):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Get the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(output_dir, latest_checkpoint)
            print(f"Resuming training from checkpoint: {checkpoint_dir}")
    
    trainer.train(resume_from_checkpoint=checkpoint_dir)

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)

    # --- Evaluation ---
    model.eval()

    print("\n--- Starting Evaluation ---")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device_map={'': 0},
        )
    
    # Prepare queries for evaluation
    eval_dataset = instructions_ds_dict['eval']
                              
    # Extract just the prompt part (up to '### Assistant:')
    queries = [
        eval_dataset['text'][i].split('### Assistant: ')[0] + '### Assistant:'
        for i in range(len(eval_dataset))
        ]
    
    print("printing first query for verification:\n",queries[0])  # Print the first query for verification#################

    # Generate predictions
    sequences = pipe(
        queries,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=3, # Max tokens needed for 'entailment', 'neutral', or 'contradiction'
        do_sample=False, # Use greedy decoding for consistent comparison
        # early_stopping=True, # No need for early stopping with small max_new_tokens
        )
    
    print('looking at sequences[0] for verification:\n',sequences[0])  # Print the first generated sequence for verification#############

    # Extract results and true labels
    results = []
    labels = []

    for i, seq in enumerate(sequences):
        # Generated text: model output after '### Assistant:'
        generated_text = seq[0]['generated_text']
        try:
            result = generated_text.split('### Assistant:')[1].strip().lower()
            results.append(result)
        except IndexError:
        # Handle cases where the model fails to generate '### Assistant:' properly
            results.append("")
        
        # True label: model output after '### Assistant:' from the original dataset
        true_label = eval_dataset['text'][i].split('### Assistant:')[1].strip().lower()
        labels.append(true_label)

        # Calculate Accuracy
        # Check if the generated result *contains* the true label (more robust than exact match)

    print("printing first 5 results and labels for verification: \n", list(zip(results[:5], labels[:5])))########################
    correct_predictions = 0

    for gen_result, true_label in zip(results, labels):
        # We check if the expected label is *in* the generated text,
        # allowing for minor differences like surrounding spaces or punctuation.
        if true_label in gen_result:
            correct_predictions += 1
    accuracy = correct_predictions / len(labels)
    print(f"Evaluation Examples: {len(labels)}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    sys.exit(int(main() or 0))