# Clinical Note Summarization

This project aims to develop a system for summarizing clinical notes using sequence-to-sequence models. This repository currently contains the first part of the project: model training and evaluation.

## Project Status

*   **Data Processing:** Completed.
*   **Model Training:** Completed for BioBART and ClinicalT5 models.
*   **Model Evaluation:** Completed, comparing the fine-tuned BioBART and ClinicalT5 models. ClinicalT5 was found to perform significantly better.

## Repository Structure

*   `data_processing.py`: Script for loading the dataset (expected in Parquet format in the `data/` directory), cleaning the text, and splitting it into training, validation, and test sets.
*   `train.py`: Script for fine-tuning sequence-to-sequence models (e.g., BioBART, ClinicalT5) on the processed data. It handles tokenization, training arguments, and saving the fine-tuned model checkpoints to the `models/` directory.
*   `run_evaluation.py`: Script for evaluating the performance of the fine-tuned models. It loads models from the `models/` directory, generates summaries for the test set, calculates ROUGE and BLEU scores, and saves the results to `evaluation_results.csv`.
*   `requirements.txt`: Lists the necessary Python packages for running the scripts.
*   `evaluation_results.csv`: Contains the evaluation metrics comparing the performance of the trained models.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore (e.g., large data files, model checkpoints).
*   `data/`: (Ignored by Git) Directory intended to hold the input dataset (e.g., `train-*.parquet`).
*   `models/`: (Ignored by Git) Directory where fine-tuned model checkpoints and training logs are saved.

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sgessesse/Clinical_Note_Summarization
    cd Clinical_Note_Summarization
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Data:** Place your clinical notes dataset (in Parquet format, e.g., `train-00000-of-00001.parquet`) inside a `data/` directory in the project root.
5.  **Run Data Processing (if needed, usually handled by training script):** The `train.py` script typically calls the data processing functions.
6.  **Run Training:** Modify `train.py` to specify the desired base model and training arguments, then run:
    ```bash
    python train.py
    ```
    *Note: Training requires significant computational resources (GPU recommended) and time.*
7.  **Run Evaluation:** After training, run the evaluation script:
    ```bash
    python run_evaluation.py
    ```
    This will generate summaries, calculate metrics, print a comparison table, and save the results to `evaluation_results.csv`.

## Evaluation Results Summary

The evaluation compared fine-tuned versions of BioBART-v2-Base and ClinicalT5-Base.

| Model                       | rouge1  | rouge2  | rougeL  | bleu    |
| :-------------------------- | :------ | :------ | :------ | :------ |
| BioBART-v2-Base (Finetuned) | 5.11    | 2.36    | 3.98    | 0.00    |
| ClinicalT5-Base (Finetuned) | 49.32   | 32.91   | 40.26   | 27.77   |

Based on these results, **ClinicalT5-Base (Finetuned)** demonstrated significantly better performance on this summarization task.

## Next Steps

*   Model deployment: Develop an application or API to serve the best-performing model (ClinicalT5) for summarizing new clinical notes.
