"""
Clinical Note Summarization Web Application

This module implements a FastAPI web application that provides an interface for summarizing
clinical notes using a fine-tuned ClinicalT5 model. The application includes both API
endpoints and a web interface for easy access.

The summarization model is based on ClinicalT5 (Lu et al., 2022) and has been fine-tuned
on a dataset of clinical notes for the specific task of summarization.
This version attempts to use GPU if available.

Author: Semir (adapted for GPU by Roo)
License: MIT
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # Restored imports
import torch # Restored import
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
    description="A web application for generating concise summaries of clinical notes using a fine-tuned ClinicalT5 model.",
    version="1.1.0" # Updated version for UI changes
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Model configuration
MODEL_PATH = "models/clinicalt5_finetuned" # Local model path
MAX_INPUT_LENGTH = 1024  # Maximum input length for the model
MAX_TARGET_LENGTH = 256  # Maximum summary length
PREFIX = "summarize: "   # T5 task prefix used during training

# Removed NEW_PROMPT_TEMPLATE as it causes leakage with ClinicalT5

# Initialize model and tokenizer
# Attempt to use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", use_fast=False)
    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval() # Set model to evaluation mode
    print("ClinicalT5 model and tokenizer initialized successfully.")
except Exception as e:
    print(f"Error initializing ClinicalT5 model or tokenizer: {e}")
    print(f"Please ensure the model exists at '{MODEL_PATH}' and the tokenizer 'luqh/ClinicalT5-base' is accessible.")
    raise

# HTML_CONTENT is no longer needed as we serve index.html directly

class SummarizationRequest(BaseModel):
    """Request model for the summarization endpoint."""
    text: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main web interface."""
    return FileResponse("app/static/index.html")

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    """
    Generate a summary for the provided clinical note using ClinicalT5.
    
    Args:
        request (SummarizationRequest): Request containing the text to summarize
        
    Returns:
        JSONResponse: Contains the generated summary
    """
    input_text = request.text or ""
    if not input_text.strip():
        return JSONResponse({"summary": "Input text is empty."})

    # Instruction for handling contradictions, to be prepended to the note
    contradiction_instruction = "Summarize the following clinical note comprehensively, covering key aspects such as assessment, plan, and follow-up instructions. When conflicting information exists, ensure that objective data (especially pharmacy records, lab results, imaging reports, EHR data, vital signs, and other verifiable clinical findings) is prioritized over subjective patient claims. If a contradiction is significant and cannot be resolved by this prioritization, note the contradiction. Clinical Note: "
    
    # Prepend instruction to the input text
    modified_input_text = contradiction_instruction + input_text
    
    try:
        # Tokenize input using the correct PREFIX for ClinicalT5
        inputs = tokenizer(
            PREFIX + modified_input_text, # Use modified input text
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        
        # Generate summary
        with torch.no_grad(): 
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4, 
                early_stopping=True
            )
        
        # Decode and format summary
        raw_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Process summary to join sentences with spaces and preserve paragraphs
        paragraphs = raw_summary.split('\n\n')
        processed_paragraphs = []
        for para_text in paragraphs:
            if para_text.strip(): # Ensure paragraph is not just whitespace
                sentences_in_para = custom_sent_tokenize(para_text)
                processed_paragraphs.append(" ".join(sentences_in_para))
        
        summary = "\n\n".join(processed_paragraphs) # Join paragraphs with double newline
        
        return JSONResponse({"summary": summary})

    except Exception as e:
        print(f"Error during ClinicalT5 summarization: {e}")
        return JSONResponse({"summary": f"Error generating summary: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    # Run app locally for testing
    print("Starting FastAPI server with Uvicorn (ClinicalT5 model)...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)