"""
Clinical Note Summarization Web Application

This module implements a FastAPI web application that provides an interface for summarizing
clinical notes using a fine-tuned ClinicalT5 model. The application includes both API
endpoints and a web interface for easy access.

The summarization model is based on ClinicalT5 (Lu et al., 2022) and has been fine-tuned
on a dataset of clinical notes for the specific task of summarization.

Author: Your Name
License: MIT
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import uvicorn
import re
import os

def custom_sent_tokenize(text):
    """
    Custom sentence tokenizer optimized for clinical text.
    
    Handles common medical abbreviations and splits text into sentences based on
    standard sentence endings followed by capital letters.
    
    Args:
        text (str): Input text to be split into sentences
        
    Returns:
        list[str]: List of sentences
    """
    # Pre-process to protect common medical abbreviations
    protected = text.replace("Dr.", "Dr[DOT]")\
                   .replace("Mr.", "Mr[DOT]")\
                   .replace("Mrs.", "Mrs[DOT]")\
                   .replace("Ms.", "Ms[DOT]")\
                   .replace("Prof.", "Prof[DOT]")\
                   .replace("vs.", "vs[DOT]")\
                   .replace("e.g.", "e.g[DOT]")\
                   .replace("i.e.", "i.e[DOT]")\
                   .replace("etc.", "etc[DOT]")\
                   .replace("approx.", "approx[DOT]")
    
    # Split on sentence endings (.!?) followed by spaces and capital letters
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    
    # Restore protected abbreviations and clean up
    sentences = [s.replace("[DOT]", ".") for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Note Summarizer",
    description="A web application for generating concise summaries of clinical notes",
    version="1.0.0"
)

# Model configuration
MODEL_PATH = "models/clinicalt5_finetuned"
MAX_INPUT_LENGTH = 1024  # Maximum input length for the model
MAX_TARGET_LENGTH = 256  # Maximum summary length
PREFIX = "summarize: "   # T5 task prefix used during training

# Initialize model and tokenizer
device = torch.device("cpu")  # Using CPU for demo deployment
tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)

# HTML template for the web interface
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Note Summarization</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 0 20px; }
        textarea { width: 100%; font-size: 1em; }
        button { padding: 10px 20px; font-size: 1em; }
        #summary-output { white-space: pre-wrap; background: #f5f5f5; padding: 10px; border-radius: 5px; min-height: 100px; }
        .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #ccc; border-top-color: #333; border-radius: 50%; animation: spin 1s linear infinite; margin-left: 10px; vertical-align: middle; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <h1>Clinical Note Summarization</h1>
    <textarea id="text-input" rows="20" placeholder="Paste clinical note here..."></textarea><br/>
    <input type="file" id="file-input" accept=".txt" style="margin-top:10px;"><br/>
    <button id="summarize-button" onclick="summarize()">Summarize</button>
    <span id="loading-indicator" style="display:none;"><div class="spinner"></div> Summarizing...</span>
    <h2>Summary</h2>
    <pre id="summary-output"></pre>

    <script>
        async function summarize() {
            const btn = document.getElementById('summarize-button');
            const loader = document.getElementById('loading-indicator');
            const output = document.getElementById('summary-output');
            btn.disabled = true;
            loader.style.display = 'inline-block';
            output.textContent = '';
            let text = document.getElementById('text-input').value;
            const fileInput = document.getElementById('file-input');
            if (fileInput.files.length > 0) {
                text = await fileInput.files[0].text();
                document.getElementById('text-input').value = text;
            }
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await response.json();
            output.textContent = data.summary;
            btn.disabled = false;
            loader.style.display = 'none';
        }
    </script>
</body>
</html>
"""

class SummarizationRequest(BaseModel):
    """Request model for the summarization endpoint."""
    text: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the web interface."""
    return HTML_CONTENT

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    """
    Generate a summary for the provided clinical note.
    
    Args:
        request (SummarizationRequest): Request containing the text to summarize
        
    Returns:
        JSONResponse: Contains the generated summary
    """
    input_text = request.text or ""
    
    # Tokenize input
    inputs = tokenizer(
        PREFIX + input_text,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(device)
    
    # Generate summary (using greedy decoding for speed)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=1,
            early_stopping=True
        )
    
    # Decode and format summary
    raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sentences = custom_sent_tokenize(raw_summary)
    summary = "\n\n".join(sentences)
    
    return JSONResponse({"summary": summary})

if __name__ == "__main__":
    # Run app locally for testing
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 