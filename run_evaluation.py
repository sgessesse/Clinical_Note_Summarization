import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoConfig
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import evaluate
import nltk
from data_processing import load_and_split_data # To get the test set

# --- Configuration ---
BIOBART_ORIGINAL_ID = "GanjinZero/biobart-v2-base"
CLINICALT5_ORIGINAL_ID = "luqh/ClinicalT5-base"
BIOBART_FINETUNED_PATH = "./models/biobart_finetuned"
CLINICALT5_FINETUNED_PATH = "./models/clinicalt5_finetuned"
MAX_INPUT_LENGTH = 1024  # Should match training
MAX_TARGET_LENGTH = 256 # Should match training
# Define batch sizes per model type
EVAL_BATCH_SIZE_BIOBART = 12
EVAL_BATCH_SIZE_CLINICALT5 = 6

# Explicitly check for CUDA and print status
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available. Using CPU.")

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
def evaluate_model(model_path: str, test_dataset: Dataset, eval_batch_size: int):
    """Loads a fine-tuned model and evaluates it on the test dataset."""
    print(f"\n--- Evaluating Model: {model_path} --- Batch Size: {eval_batch_size} ---")

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    try:
        # Determine if it's a T5 model
        is_t5_model = "t5" in model_path.lower()
        prefix = "summarize: " if is_t5_model else ""

        # Load tokenizer first
        print(f"Loading tokenizer for {model_path}...")
        # Still need use_fast=False if sentencepiece is not installed
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        # Load model
        print(f"Loading model {model_path}...")
        if model_path == CLINICALT5_ORIGINAL_ID:
            # Manual loading: Config -> Empty PyTorch Model -> Load Flax Weights
            print("Attempting manual load: Config -> PyTorch structure -> Load Flax weights...")
            print(f"Target device for loading: {DEVICE}") # Print target device

            print("Loading ClinicalT5 config...")
            config = AutoConfig.from_pretrained(model_path)

            print("Initializing PyTorch model structure...")
            model = AutoModelForSeq2SeqLM.from_config(config)
            # Check device immediately after initialization
            print(f"  Model device after from_config: {model.device}")

            print("Downloading Flax checkpoint...")
            flax_checkpoint_path = hf_hub_download(repo_id=model_path, filename="flax_model.msgpack")

            print("Loading Flax weights into PyTorch model...")
            model = load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path)
            # Check device after loading flax weights
            print(f"  Model device after load_flax_checkpoint: {model.device}")

            print("Moving model to device...")
            model.to(DEVICE)
            # Add check for model device after moving
            print(f"  Model device after model.to(DEVICE): {model.device}")
            print("Successfully loaded ClinicalT5 model manually.")
        else:
            # Standard loading for PyTorch models
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            model.to(DEVICE)

        model.eval() # Set model to evaluation mode
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model/tokenizer at {model_path}: {e}")
        # If sentencepiece is missing, this might still fail here for the tokenizer
        if "SentencePiece" in str(e):
             print("\n*** ERROR: SentencePiece library not found. This is required for the ClinicalT5 tokenizer. ***")
             print("*** Evaluation results for ClinicalT5 will be unreliable without the correct tokenizer. ***")
             print("*** Please install SentencePiece: https://github.com/google/sentencepiece#installation ***")
        return None

    # 2. Preprocess Test Data
    print("Tokenizing test data inputs...")
    def tokenize_inputs(examples):
        # Add prefix for T5 models
        inputs = [prefix + doc for doc in examples["text"]]
        
        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return model_inputs

    tokenized_inputs = test_dataset.map(
        tokenize_inputs,
        batched=True,
        remove_columns=["text", "summary"]
    )
    tokenized_inputs.set_format("torch")
    print("Tokenization complete.")

    # 3. DataLoader setup
    eval_dataloader = DataLoader(tokenized_inputs, batch_size=eval_batch_size)

    # 4. Generate Summaries
    print(f"Generating summaries on device: {DEVICE}...")
    all_preds = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader, desc="Generating")):
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Print device for the first batch only
            if i == 0:
                print(f"  Input batch tensors are on device: {batch['input_ids'].device}")

            # Generate with appropriate parameters for clinical summarization
            generated_ids = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=MAX_TARGET_LENGTH,
                min_length=50,  # Increased for clinical summaries
                num_beams=4,
                length_penalty=1.0,  # Balanced length penalty
                no_repeat_ngram_size=3,  # Avoid repetition
                early_stopping=True,
                do_sample=False,  # Deterministic generation
                temperature=1.0,  # No temperature scaling
                top_p=1.0  # No nucleus sampling
            )
            
            # Decode predictions
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_preds.extend(preds)
    
    print("Summary generation complete.")

    # 5. Post-process and Calculate Metrics
    print("Calculating metrics...")
    # Get original summaries
    original_summaries = test_dataset["summary"]
    
    # Post-processing
    decoded_preds = [pred.strip() for pred in all_preds]
    decoded_labels = [label.strip() for label in original_summaries]
    decoded_labels_bleu = [[label] for label in decoded_labels]

    # ROUGE expects newline separation
    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    # Compute ROUGE
    rouge_result = rouge_metric.compute(
        predictions=decoded_preds_rouge,
        references=decoded_labels_rouge,
        use_stemmer=True
    )
    rouge_result = {key: value * 100 for key, value in rouge_result.items()}

    # Compute BLEU
    bleu_result = bleu_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels_bleu,
        force=True # Suppress warnings about tokenized periods
    )
    bleu_result = {"bleu": bleu_result["score"]}

    # Combine metrics
    result = {**rouge_result, **bleu_result}

    # Add mean generated length
    prediction_lens = [len(tokenizer.encode(pred)) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)

    final_metrics = {k: round(v, 4) for k, v in result.items()}
    print("Metrics calculation complete.")
    print(final_metrics)

    # Save some example predictions for manual inspection
    print("\nSaving example predictions for manual inspection...")
    examples_df = pd.DataFrame({
        'Original Text': test_dataset['text'][:5],
        'Original Summary': test_dataset['summary'][:5],
        'Generated Summary': decoded_preds[:5]
    })
    examples_df.to_csv(f"example_predictions_{model_path.replace('/', '_')}.csv", index=False)
    print("Example predictions saved.")

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
        # Original models
        {"name": "BioBART-v2-Base (Original)", "path": BIOBART_ORIGINAL_ID, "batch_size": EVAL_BATCH_SIZE_BIOBART},
        {"name": "ClinicalT5-Base (Original)", "path": CLINICALT5_ORIGINAL_ID, "batch_size": EVAL_BATCH_SIZE_CLINICALT5},
        # Finetuned models
        {"name": "BioBART-v2-Base (Finetuned)", "path": BIOBART_FINETUNED_PATH, "batch_size": EVAL_BATCH_SIZE_BIOBART},
        {"name": "ClinicalT5-Base (Finetuned)", "path": CLINICALT5_FINETUNED_PATH, "batch_size": EVAL_BATCH_SIZE_CLINICALT5},
    ]

    # 3. Run Evaluation
    results = {}
    # First load existing results from CSV if it exists
    if os.path.exists("evaluation_results.csv"):
        print("Loading existing results from evaluation_results.csv")
        existing_results = pd.read_csv("evaluation_results.csv", index_col=0)
        results = existing_results.to_dict('index')
    else:
        print("No existing evaluation_results.csv found.")

    for model_info in models_to_evaluate:
        metrics = evaluate_model(
            model_info["path"],
            test_dataset,
            model_info["batch_size"] # Pass the specific batch size
        )
        if metrics:
            results[model_info["name"]] = metrics

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
