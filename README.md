# Clinical Note Summarization

A web application that automatically generates concise summaries of clinical notes using state-of-the-art transformer models fine-tuned on clinical text data.

## Live Demo

The application is deployed and accessible at: [https://clinical-summarizer-995990015160.us-central1.run.app/](https://clinical-summarizer-995990015160.us-central1.run.app/)

**Note**: The current deployment is CPU-only for demonstration purposes, resulting in ~20 second summarization time. In a production environment with GPU support, summarization would take <1-2 seconds.

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
- **Deployment**: Google Cloud Run (4GB RAM provisioned)
- **Container**: Docker

## Project Structure

- `data_processing.py`: Script for loading and preprocessing the clinical notes dataset
- `train.py`: Script for fine-tuning transformer models (BioBART and ClinicalT5)
- `run_evaluation.py`: Script for evaluating model performance using ROUGE and BLEU metrics
- `app/`: Directory containing the FastAPI web application
- `models/`: Directory for storing fine-tuned model checkpoints (not included in repository)
- `requirements.txt`: Python package dependencies
- `Dockerfile`: Container configuration for deployment
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

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|-------|---------|---------|---------|------|
| BioBART-v2-Base (Finetuned) | 5.11 | 2.36 | 3.98 | 0.00 |
| ClinicalT5-Base (Finetuned) | 49.32 | 32.91 | 40.26 | 27.77 |

Based on these results, the fine-tuned ClinicalT5 model significantly outperformed BioBART and was selected for deployment in the web application.

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

## Docker Deployment

1. Build the container:
```bash
docker build -t clinical-summarizer .
```

2. Run locally:
```bash
docker run -p 8080:8080 clinical-summarizer
```

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
