import os
import nltk
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoConfig
)
from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model # Import specific utility
from huggingface_hub import hf_hub_download # Import download utility
import evaluate
import re
from data_processing import load_and_split_data # Import our data loading function

# --- Configuration ---
BIOBART_MODEL_ID = "GanjinZero/biobart-v2-base"
CLINICALT5_MODEL_ID = "luqh/ClinicalT5-base"
MAX_INPUT_LENGTH = 1024  # Max length for input text
MAX_TARGET_LENGTH = 256 # Max length for generated summary
TRAIN_BATCH_SIZE = 2   # Reduced from 8 for T5
EVAL_BATCH_SIZE = 2     # Reduced from 8 for T5
GRAD_ACCUM_STEPS = 4     # Gradient accumulation steps
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5           # Changed from 3 to 5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
OUTPUT_DIR_BIOBART = "./models/biobart_finetuned"
OUTPUT_DIR_CLINICALT5 = "./models/clinicalt5_finetuned"
METRIC_FOR_BEST_MODEL = "rougeL" # Metric to determine the best checkpoint

# --- Setup ---

# Download nltk punkt tokenizer data (needed for rouge_score)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Correct exception type
    print("Downloading nltk punkt tokenizer...")
    nltk.download('punkt', quiet=True)
# Also try downloading punkt_tab needed by sent_tokenize in some cases
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading nltk punkt_tab resource...")
    nltk.download('punkt_tab', quiet=True)

# Load evaluation metrics
print("Loading evaluation metrics (ROUGE and BLEU)...")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("sacrebleu")
print("Metrics loaded.")

# Global variables for tokenizer and prefix (set within training loop)
tokenizer = None
prefix = ""

# --- Preprocessing Function ---
def preprocess_function(examples):
    """Tokenizes inputs and labels for seq2seq models."""
    global tokenizer, prefix, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    if tokenizer is None:
        raise ValueError("Tokenizer not set globally before preprocessing.")

    # Prepend prefix if needed (for T5)
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")

    # Setup the tokenizer for targets (labels)
    labels = tokenizer(text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Compute Metrics Function ---
def compute_metrics(eval_pred):
    """Computes ROUGE and BLEU scores for evaluation."""
    global tokenizer, rouge_metric, bleu_metric
    if tokenizer is None:
        raise ValueError("Tokenizer not set globally before computing metrics.")

    predictions, labels = eval_pred
    # Predictions are often logit tuples, take the first element if needed
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Replace -100 in labels used for padding/masking
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing: remove leading/trailing spaces
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    decoded_labels_bleu = [[label] for label in decoded_labels] # BLEU expects list of references

    # ROUGE expects newline separation for sentences
    decoded_preds_rouge = ["\n".join(nltk.sent_tokenize(pred)) for pred in decoded_preds]
    decoded_labels_rouge = ["\n".join(nltk.sent_tokenize(label)) for label in decoded_labels]

    # Compute ROUGE
    rouge_result = rouge_metric.compute(predictions=decoded_preds_rouge, references=decoded_labels_rouge, use_stemmer=True)
    rouge_result = {key: value * 100 for key, value in rouge_result.items()} # Use score or fmeasure

    # Compute BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels_bleu)
    bleu_result = {"bleu": bleu_result["score"]}

    # Combine metrics
    result = {**rouge_result, **bleu_result}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# --- Main Training Function ---
def train_model(model_id, output_dir, raw_datasets):
    """Loads, preprocesses, trains, and saves a seq2seq model."""
    global tokenizer, prefix # Allow modification of global tokenizer/prefix

    print(f"\n--- Training Model: {model_id} ---")
    print(f"Output directory: {output_dir}")

    # 1. Load Tokenizer and Model
    print("Loading tokenizer and model...")
    # Load tokenizer first based on model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set prefix and load model based on type
    if model_id == CLINICALT5_MODEL_ID:
        prefix = "summarize: "
        print(f"Loading {model_id} config...")
        config = AutoConfig.from_pretrained(model_id)
        print(f"Initializing PyTorch model structure for {model_id}...")
        # Initialize empty PyTorch model from config
        model = AutoModelForSeq2SeqLM.from_config(config)
        print(f"Downloading Flax checkpoint file for {model_id}...")
        # Ensure Flax model is downloaded and get the path to the checkpoint file
        flax_checkpoint_path = hf_hub_download(repo_id=model_id, filename="flax_model.msgpack")
        print(f"Loading Flax weights from {flax_checkpoint_path} into PyTorch model...")
        # Load Flax weights using the file path
        model = load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path)
        print("Successfully loaded model from Flax weights into PyTorch structure.")

    elif "t5" in model_id.lower():
        # Handle other T5 models (if any added later) - Standard PyTorch load
        prefix = "summarize: "
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id) # Standard PyTorch load
    elif model_id == BIOBART_MODEL_ID:
         # Handle BioBART - Standard PyTorch load
         prefix = ""
         model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    else:
         # Default loading for any other models - Standard PyTorch load
         prefix = "" # Assuming no prefix needed by default
         model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    print("Tokenizer and model loaded.")

    # 2. Preprocess Data
    print("Preprocessing datasets...")
    # Ensure previous tokenizer/prefix are cleared if reusing function
    # (Handled by setting tokenizer globally within this function)
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names # Remove all original columns
    )
    print("Preprocessing complete.")
    print(tokenized_datasets)

    # 3. Data Collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # 4. Training Arguments
    # Check BF16 availability
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if use_bf16:
        print("BF16 is available and will be used.")
    else:
        print("BF16 not available on this hardware/setup. Consider FP16 if needed.")
        # Optionally fallback to FP16 if BF16 isn't supported but FP16 is desired
        # use_fp16 = torch.cuda.is_available() # Basic check for CUDA

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch", # Changed from evaluation_strategy
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        bf16=use_bf16,
        fp16=False, # Explicitly disable FP16 if using BF16
        report_to="tensorboard",
        push_to_hub=False,
        save_total_limit=2, # Keep only the best and latest checkpoints
    )
    print("Training arguments configured.")

    # 5. Trainer Initialization
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("Trainer initialized.")

    # 6. Start Training
    print(f"Starting training for {model_id}...")
    try:
        train_result = trainer.train()
        print("Training finished.")

        # 7. Save Best Model & Metrics
        print("Saving best model, tokenizer, and metrics...")
        trainer.save_model()  # Saves the best model based on metric_for_best_model
        tokenizer.save_pretrained(output_dir)
        print(f"Best model and tokenizer saved to {output_dir}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Training metrics and state saved.")

    except Exception as e:
        print(f"An error occurred during training for {model_id}: {e}")
        # Consider logging the error or raising it depending on desired behavior

    finally:
        # Clean up globals to prevent interference between model runs
        tokenizer = None
        prefix = ""
        # Optionally clear CUDA cache if memory issues arise between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Finished processing model: {model_id}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CUDA Check ---
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Current device index: {torch.cuda.current_device()}")
    else:
        print("CUDA is NOT available. Training will run on CPU.")
    # --- End CUDA Check ---

    print("\nStarting the training pipeline...")

    # Load and split data once
    print("Loading and splitting data...")
    raw_datasets = load_and_split_data()
    print("Data loading and splitting complete.")

    # Define models and output directories
    models_to_train = [
        # {"id": BIOBART_MODEL_ID, "output_dir": OUTPUT_DIR_BIOBART}, # Already trained
        {"id": CLINICALT5_MODEL_ID, "output_dir": OUTPUT_DIR_CLINICALT5},
    ]

    # Train each model sequentially
    for model_info in models_to_train:
        train_model(
            model_id=model_info["id"],
            output_dir=model_info["output_dir"],
            raw_datasets=raw_datasets # Pass the already loaded and split data
        )

    print("\n--- All Training Runs Complete ---")
