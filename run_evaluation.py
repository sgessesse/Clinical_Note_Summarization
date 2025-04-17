import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import nltk
from data_processing import load_and_split_data # To get the test set

# --- Configuration ---
BIOBART_FINETUNED_PATH = "./models/biobart_finetuned"
CLINICALT5_FINETUNED_PATH = "./models/clinicalt5_finetuned"
MAX_INPUT_LENGTH = 1024  # Should match training
MAX_TARGET_LENGTH = 256 # Should match training
EVAL_BATCH_SIZE = 8      # Can be larger for inference if VRAM allows
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
# Ensure nltk punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Correct exception type
    print("Downloading nltk punkt tokenizer...")
    nltk.download('punkt')

# Load evaluation metrics
print("Loading evaluation metrics (ROUGE and BLEU)...")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("sacrebleu")
print("Metrics loaded.")

# --- Helper Function for Evaluation ---
def evaluate_model(model_path: str, test_dataset: Dataset):
    """Loads a fine-tuned model and evaluates it on the test dataset."""
    print(f"\n--- Evaluating Model: {model_path} ---")

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model and tokenizer loaded.")
    except OSError:
        print(f"Error: Could not find saved model/tokenizer at {model_path}. Did training complete successfully?")
        return None

    # Determine prefix based on model type (heuristic)
    prefix = "summarize: " if "t5" in model_path.lower() else ""

    # 2. Preprocess Test Data (Tokenization only for input)
    def tokenize_inputs(examples):
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length", return_tensors="pt")
        return model_inputs

    print("Tokenizing test data inputs...")
    # We only need to tokenize inputs for generation; keep original summaries for reference
    tokenized_inputs = test_dataset.map(lambda x: tokenize_inputs(x), batched=True, remove_columns=["text", "summary"])
    tokenized_inputs.set_format("torch")

    # Keep original summaries for comparison
    original_summaries = test_dataset["summary"]

    # 3. DataLoader
    eval_dataloader = DataLoader(tokenized_inputs, batch_size=EVAL_BATCH_SIZE)

    # 4. Generate Summaries
    print(f"Generating summaries using device: {DEVICE}...")
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Generating"):
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # Generate
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_TARGET_LENGTH,
                num_beams=4, # Example beam search configuration
                early_stopping=True
            )
            # Decode and store
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_preds.extend(preds)
    print("Summary generation complete.")

    # 5. Post-process and Calculate Metrics
    print("Calculating metrics...")
    # Simple post-processing
    decoded_preds = [pred.strip() for pred in all_preds]
    decoded_labels = [label.strip() for label in original_summaries]
    decoded_labels_bleu = [[label] for label in decoded_labels] # BLEU expects list of references

    # ROUGE expects newline separation
    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    # Compute ROUGE
    rouge_result = rouge_metric.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
    rouge_result = {key: value * 100 for key, value in rouge_result.items()}

    # Compute BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    bleu_result = {"bleu": bleu_result["score"]}

    # Combine metrics
    result = {**rouge_result, **bleu_result}

    # Add mean generated length
    prediction_lens = [len(tokenizer.encode(pred, add_special_tokens=False)) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    final_metrics = {k: round(v, 4) for k, v in result.items()}
    print("Metrics calculation complete.")
    print(final_metrics)

    return final_metrics

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting evaluation pipeline...")

    # 1. Load Data (only need the test split)
    print("Loading and splitting data to get the test set...")
    try:
        raw_datasets = load_and_split_data()
        test_dataset = raw_datasets['test']
        print(f"Test set loaded: {len(test_dataset)} records.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        exit(1)

    # 2. Define Models to Evaluate
    models_to_evaluate = [
        {"name": "BioBART-v2-Base (Finetuned)", "path": BIOBART_FINETUNED_PATH},
        {"name": "ClinicalT5-Base (Finetuned)", "path": CLINICALT5_FINETUNED_PATH},
    ]

    # 3. Run Evaluation
    results = {}
    for model_info in models_to_evaluate:
        # Check if model directory exists before attempting evaluation
        if os.path.isdir(model_info["path"]):
            metrics = evaluate_model(model_info["path"], test_dataset)
            if metrics:
                results[model_info["name"]] = metrics
        else:
            print(f"Skipping evaluation for {model_info['name']}: Directory not found at {model_info['path']}")

    # 4. Display Results
    print("\n--- Evaluation Results Summary ---")
    if results:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        print(results_df.to_markdown()) # Print as markdown table

        # Optionally save results
        results_df.to_csv("evaluation_results.csv")
        print("\nResults saved to evaluation_results.csv")

        # Determine best model based on ROUGE-L
        METRIC_FOR_BEST_MODEL = 'rougeL' # Define the metric to use for comparison
        best_model_name = results_df[METRIC_FOR_BEST_MODEL].idxmax() # Use the exact column name
        print(f"\nBest model based on {METRIC_FOR_BEST_MODEL}: {best_model_name}")

    else:
        print("No models were successfully evaluated.")

    print("\n--- Evaluation Complete ---")
