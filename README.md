# Clinical Note Summarization

A web application that automatically generates concise summaries of clinical notes using state-of-the-art transformer models fine-tuned on clinical text data.

## Live Demo

The application is deployed and accessible at: [https://sem-werede--clinical-summarizer-app-web-app.modal.run](https://sem-werede--clinical-summarizer-app-web-app.modal.run)

**Note**: The current deployment leverages serverless GPUs via Modal, resulting in an average summarization time of ~3 seconds (a significant improvement from the previous ~20 seconds on CPU).

## Features

- Web-based interface for easy input of clinical notes
- Support for both text input and file upload
- Real-time summarization with progress indicator
- Responsive design that works on desktop and mobile devices
- Automatic sentence segmentation for improved readability

## Technical Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript
- **Model**: Fine-tuned ClinicalT5 for summarization
- **Deployment**: Modal (Serverless GPU)
- **Container**: Docker

## Project Structure

- `data_processing.py`: Script for loading and preprocessing the clinical notes dataset
- `train.py`: Script for fine-tuning transformer models (BioBART and ClinicalT5)
- `run_evaluation.py`: Script for evaluating model performance using ROUGE and BLEU metrics
- `app/`: Directory containing the FastAPI web application
- `models/`: Directory for storing fine-tuned model checkpoints (not included in repository)
- `requirements.txt`: Python package dependencies
- `Dockerfile`: Original container configuration (primarily for local Docker testing or alternative deployments)
- `modal_deploy.py`: Script for deploying the application to Modal with serverless GPU.
- `evaluation_results.csv`: Model evaluation metrics

## Model Training and Evaluation

We fine-tuned two transformer models on the clinical notes summarization task:
1. BioBART-v2-Base
2. ClinicalT5-Base

The training process involved:
1. Data preprocessing using `data_processing.py`
2. Model fine-tuning using `train.py`
3. Performance evaluation using `run_evaluation.py`

### Evaluation Results

| Model                         | rouge1  | rouge2  | rougeL  | rougeLsum | bleu    | gen_len  |
| :---------------------------- | :------ | :------ | :------ | :-------- | :------ | :------- |
| BioBART-v2-Base (Original)  | 42.1308 | 23.5126 | 28.4806 | 38.7688   | 13.14   | 254.4863 |
| BioBART-v2-Base (Finetuned) | 58.823  | 39.6999 | 47.2348 | 54.9708   | 34.4715 | 115.5956 |
| ClinicalT5-Base (Finetuned) | 60.2571 | 42.1793 | 49.6858 | 56.5992   | 37.1784 | 132.3122 |
| ClinicalT5-Base (Original)  | 16.2759 | 5.841   | 12.1763 | 14.5375   | 1.1683  | 45.0182  |

Based on these results, the fine-tuned ClinicalT5 model significantly outperformed the other models and was selected for deployment in the web application.

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/sgessesse/Clinical_Note_Summarization.git
cd Clinical_Note_Summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python -m uvicorn app.main:app --reload
```

4. Visit `http://localhost:8000` in your browser

## Modal Deployment

The application is deployed using [Modal](https://modal.com) for serverless GPU execution.

1. **Install Modal Client**:
   ```bash
   pip install modal
   ```
2. **Setup Modal Token**:
   ```bash
   modal setup
   ```
   (This will open a browser for authentication)
3. **Deploy the Application**:
   Navigate to the project root directory and run:
   ```bash
   modal deploy modal_deploy.py
   ```
   Modal will provide a URL for the deployed application.

## Local Docker Testing (Optional)

If you wish to run the application locally using Docker (without Modal's serverless GPU benefits):

1. Build the container (ensure `Dockerfile` is configured for CPU execution if not using a GPU-enabled Docker setup):
```bash
docker build -t clinical-summarizer .
```

2. Run locally:
```bash
docker run -p 8000:8000 clinical-summarizer
```
(Note: The port in `Dockerfile` might need to be 8000 to match `app/main.py`'s Uvicorn default if not overridden)

## Training Your Own Model

Due to size constraints, the fine-tuned models are not included in this repository. To train your own models:

1. Download the dataset from [ayush0205/clinical_notes_renamed](https://huggingface.co/datasets/ayush0205/clinical_notes_renamed)
2. Process the data:
```bash
python data_processing.py
```
3. Fine-tune the models:
```bash
python train.py
```
4. Evaluate performance:
```bash
python run_evaluation.py
```
5. Place the best performing model in the `models/` directory

## Base Models

This project uses the following models as starting points for fine-tuning:

- [ClinicalT5](https://huggingface.co/luqh/ClinicalT5-base) (Lu et al., 2022)
- [BioBART](https://arxiv.org/abs/2204.03905) (Yuan et al., 2022)

```bibtex
@inproceedings{lu-etal-2022-clinicalt5,
    title = "Clinical{T}5: A Generative Language Model for Clinical Text",
    author = "Lu, Qiuhao and Dou, Dejing and Nguyen, Thien",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics",
    pages = "5436--5443",
    address = "Abu Dhabi, United Arab Emirates"
}

@misc{BioBART,
  title={BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model},
  author={Hongyi Yuan and Zheng Yuan and Ruyi Gan and Jiaxing Zhang and Yutao Xie and Sheng Yu},
  year={2022},
  eprint={2204.03905},
  archivePrefix={arXiv}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
