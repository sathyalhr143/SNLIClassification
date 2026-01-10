# SNLI Text Classification with Llama 3.1 8B

Fine-tuned Meta-Llama-3.1-8B-Instruct model for Natural Language Inference (NLI) classification on the Stanford Natural Language Inference (SNLI) dataset.

## 🎯 Model Performance

- **Overall Accuracy**: 91.57%
- **Test Examples**: 9,824
- **Correct Predictions**: 8,996

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Entailment** | 91.76% | 92.58% | 92.17% | 3,368 |
| **Neutral** | 88.65% | 87.36% | 88.00% | 3,219 |
| **Contradiction** | 94.22% | 94.72% | 94.47% | 3,237 |

### Confusion Matrix

|  | Predicted: Entailment | Predicted: Neutral | Predicted: Contradiction |
|---|---|---|---|
| **True: Entailment** | 3,118 | 217 | 33 |
| **True: Neutral** | 252 | 2,812 | 155 |
| **True: Contradiction** | 28 | 143 | 3,066 |

## 🚀 Quick Start

### Prerequisites

```bash
conda create -n ml python=3.10
conda activate ml
pip install torch transformers datasets peft trl accelerate bitsandbytes wandb scikit-learn matplotlib seaborn python-dotenv
```

### Setup

1. Clone this repository
2. Create a `.env` file with your Hugging Face token:
   ```
   HF_TOKEN=your_token_here
   ```

### Training

```bash
python SNLIClassification.py
```

**Training Configuration**:
- Base Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- LoRA Config: r=64, alpha=16, target_modules="all-linear"
- Quantization: 4-bit
- Batch Size: 4
- Learning Rate: 1.41e-5
- Epochs: 1 (increase to 3 for better results)

### Testing

```bash
python model_testing.py
```

Evaluates the model on the SNLI test set and generates:
- `test_results.txt`: Summary metrics
- `detailed_predictions.txt`: All predictions with correctness labels

### Error Analysis

```bash
python error_analysis.py
```

Generates:
- Detailed breakdown of error types
- Text length correlation analysis
- Sample error cases with full text
- `error_analysis_report.txt`: Complete error report

### Visualizations

```bash
python visualize_results.py
```

Creates visualizations:
- `confusion_matrix.png`: Confusion matrix heatmap
- `class_performance.png`: Per-class accuracy bar chart
- `error_distribution.png`: Error type distribution
- `overall_metrics.png`: Overall performance summary

## 📁 Project Structure

```
SNLIClassification/
├── SNLIClassification.py      # Training script
├── model_testing.py            # Evaluation script
├── error_analysis.py           # Error analysis tool
├── visualize_results.py        # Visualization generator
├── logger.py                   # Logging utilities
├── .env                        # Environment variables (not tracked)
├── checkpoints/                # Saved model weights
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── checkpoint-68671/       # Training checkpoint
├── test_results.txt            # Test metrics summary
├── detailed_predictions.txt    # All test predictions
├── error_analysis_report.txt   # Detailed error analysis
└── *.png                       # Generated visualizations
```

## 🔧 Model Details

### Architecture
- **Base**: Meta-Llama-3.1-8B-Instruct
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Task**: Natural Language Inference (3-way classification)

### Dataset
- **Name**: Stanford Natural Language Inference (SNLI)
- **Train**: 550,152 examples
- **Validation**: 10,000 examples
- **Test**: 9,824 examples (after filtering -1 labels)

### Prompt Format

```
### Human: Classify the relationship between the following two sentences as one of the following: entailment, neutral, contradiction.
premise: [PREMISE TEXT]
hypothesis: [HYPOTHESIS TEXT]

### Assistant: [LABEL]
```

## 📊 Results Analysis

### Key Findings

1. **Strongest Performance**: Contradiction classification (94.47% F1)
2. **Weakest Performance**: Neutral classification (88.00% F1)
3. **Most Common Error**: Neutral misclassified as Entailment (252 cases)
4. **Error Rate**: 8.43% (828 errors out of 9,824 examples)

### Error Patterns

The model occasionally confuses:
- Neutral statements with Entailment (when there's semantic overlap)
- Neutral statements with Contradiction (when there's mild inconsistency)
- Entailment with Neutral (when the relationship is subtle)

## 🛠️ Configuration Options

### Checkpoint Selection

Change the checkpoint path in `model_testing.py`:
```python
checkpoint_path = "checkpoints/checkpoint-68671"  # Use specific checkpoint
```

### Batch Size Adjustment

Modify batch size based on GPU memory:
```python
# In model_testing.py, line 122
predictions, true_labels = evaluate_model(model, tokenizer, test_data, batch_size=4)
```

### Quantization Options

Adjust in both training and testing scripts:
```python
load_in_8bit = False  # For 8-bit quantization
load_in_4bit = True   # For 4-bit quantization (recommended)
```

## 🎓 Usage Examples

### Interactive Testing

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "checkpoints")
tokenizer = AutoTokenizer.from_pretrained("checkpoints")

# Test example
premise = "A man is playing guitar on stage."
hypothesis = "A person is performing music."

prompt = f"""### Human: Classify the relationship between the following two sentences as one of the following: entailment, neutral, contradiction.
premise: {premise}
hypothesis: {hypothesis}

### Assistant:"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=5)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result.split("### Assistant:")[-1].strip())
# Output: entailment
```

## 📈 Future Improvements

1. **Training**:
   - Increase epochs to 3
   - Experiment with larger LoRA rank (r=128)
   - Try different learning rate schedules

2. **Data**:
   - Add data augmentation
   - Fine-tune on MultiNLI for domain adaptation
   - Handle neutral class imbalance

3. **Architecture**:
   - Try full fine-tuning on smaller models
   - Experiment with different base models
   - Ensemble multiple checkpoints

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{snli-llama-finetuning,
  author = {Your Name},
  title = {SNLI Classification with Llama 3.1 8B},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/SNLIClassification}
}
```

### Dataset Citation

```bibtex
@inproceedings{snli:emnlp2015,
  Author = {Bowman, Samuel R. and Angeli, Gabor and Potts, Christopher and Manning, Christopher D.},
  Booktitle = {Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  Publisher = {Association for Computational Linguistics},
  Title = {A large annotated corpus for learning natural language inference},
  Year = {2015}
}
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Notes

- Requires CUDA-capable GPU with at least 8GB VRAM
- Training takes approximately 4-6 hours on a single GPU
- Testing takes approximately 1-2 hours on 9,824 examples
- Ensure sufficient disk space for model checkpoints (~20GB)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Built with ❤️ using Hugging Face Transformers, PEFT, and TRL**
