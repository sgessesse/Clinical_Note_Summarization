# modal_deploy.py

import modal
import os
import re # For custom_sent_tokenize

# Define the Modal App
app = modal.App(name="clinical-summarizer-app")

# --- GPU Image Definition (for ClinicalT5Model) ---
gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install_from_requirements("requirements.txt", force_build=True) # requirements.txt should have all ML deps
    .add_local_dir("app", remote_path="/root/app")  # For custom_sent_tokenize if used by model
    .add_local_dir("models/clinicalt5_finetuned", remote_path="/root/models/clinicalt5_finetuned")
)

# --- Web Image Definition (for FastAPI app) ---
web_image = (
    modal.Image.debian_slim()  # Use default Python in debian_slim
    .pip_install(["fastapi", "uvicorn[standard]", "pydantic"], force_build=True)
    .add_local_dir("app", remote_path="/root/app") # For static files and FastAPI app structure
)

# --- Configuration Constants (paths inside the Modal container) ---
# Note: These paths are relative to /root/ in both images due to add_local_dir
MODEL_PATH_IN_CONTAINER = "/root/models/clinicalt5_finetuned"
STATIC_FILES_PATH_IN_CONTAINER = "/root/app/static"
INDEX_HTML_PATH_IN_CONTAINER = "/root/app/static/index.html"

# --- Helper Function (from your app/main.py) ---
# This function is now part of the 'app' directory copied into both images.
# If only used by the model, it only strictly needs to be in gpu_image.
# If used by web_app (e.g. for request processing), it needs to be in web_image.
# For now, it's available in both due to copying the whole 'app' dir.
def custom_sent_tokenize(text: str) -> list[str]:
    """
    Custom sentence tokenizer optimized for clinical text.
    (Copied from your app/main.py)
    """
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
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    sentences = [s.replace("[DOT]", ".") for s in sentences]
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

# --- GPU-Powered Model Class ---
@app.cls(image=gpu_image, gpu="A10G") # Use the dedicated GPU image
class ClinicalT5Model:
    def __init__(self):
        """
        This method runs once when a new container for this class is started.
        It loads the model onto the GPU.
        """
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        self.device = torch.device("cuda") # We've requested a GPU
        print(f"ClinicalT5Model: Initializing on device: {self.device}")

        try:
            # Tokenizer can be loaded from Hugging Face hub or local path if included in MODEL_PATH_IN_CONTAINER
            self.tokenizer = AutoTokenizer.from_pretrained("luqh/ClinicalT5-base", use_fast=False)
            print(f"ClinicalT5Model: Loading model from: {MODEL_PATH_IN_CONTAINER}")
            # Ensure MODEL_PATH_IN_CONTAINER has all necessary files (pytorch_model.bin, config.json, etc.)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_IN_CONTAINER).to(self.device)
            self.model.eval() # Set model to evaluation mode
            print("ClinicalT5Model: Model and tokenizer initialized successfully.")
        except Exception as e:
            print(f"ClinicalT5Model: Critical error initializing model or tokenizer: {e}")
            # This will cause container startup to fail, which is appropriate.
            raise

        # Store constants from your app/main.py
        self.MAX_INPUT_LENGTH = 1024
        self.MAX_TARGET_LENGTH = 256
        self.PREFIX = "summarize: "
        # Instruction for handling contradictions
        self.contradiction_instruction = (
            "Summarize the following clinical note comprehensively, covering key aspects "
            "such as assessment, plan, and follow-up instructions. When conflicting "
            "information exists, ensure that objective data (especially pharmacy records, "
            "lab results, imaging reports, EHR data, vital signs, and other verifiable "
            "clinical findings) is prioritized over subjective patient claims. If a "
            "contradiction is significant and cannot be resolved by this prioritization, "
            "note the contradiction. Clinical Note: "
        )

    @modal.method() # This makes the method callable remotely
    def summarize(self, text: str) -> str:
        """
        Performs summarization on the input text using the GPU-loaded model.
        """
        import torch # Good practice to ensure torch is in method's scope if needed

        print(f"ClinicalT5Model: Received text for summarization (length: {len(text)})")
        if not text.strip():
            return "Input text is empty."

        modified_input_text = self.contradiction_instruction + text

        try:
            inputs = self.tokenizer(
                self.PREFIX + modified_input_text,
                max_length=self.MAX_INPUT_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad(): # Important for inference
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.MAX_TARGET_LENGTH,
                    num_beams=4,
                    early_stopping=True
                )

            raw_summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Process summary (same logic as your app/main.py)
            paragraphs = raw_summary.split('\n\n')
            processed_paragraphs = []
            for para_text in paragraphs:
                if para_text.strip():
                    sentences_in_para = custom_sent_tokenize(para_text)
                    processed_paragraphs.append(" ".join(sentences_in_para))
            summary = "\n\n".join(processed_paragraphs)

            print(f"ClinicalT5Model: Generated summary (length: {len(summary)})")
            return summary

        except Exception as e:
            print(f"ClinicalT5Model: Error during summarization: {e}")
            # Return an error message, or raise a more specific exception
            return f"Error generating summary: {str(e)}"

# --- FastAPI App Definition (to be served by Modal) ---
import sys
import os
print("---- MODAL CONTAINER PYTHON DIAGNOSTICS ----")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"sys.path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
# Attempt to list site-packages, common locations
try:
    # For Python 3.11 in a standard Linux env (like the nvidia/cuda base)
    site_packages_path = "/usr/local/lib/python3.11/site-packages"
    if os.path.exists(site_packages_path):
        print(f"Contents of {site_packages_path}: {os.listdir(site_packages_path)}")
    else:
        print(f"{site_packages_path} does not exist.")
    # Another common path
    alt_site_packages_path = "/usr/lib/python3/dist-packages" # Common on Debian/Ubuntu
    if os.path.exists(alt_site_packages_path):
         print(f"Contents of {alt_site_packages_path}: {os.listdir(alt_site_packages_path)}")
    else:
        print(f"{alt_site_packages_path} does not exist.")

except Exception as e_diag:
    print(f"Error during diagnostics: {e_diag}")
print("---- END MODAL CONTAINER PYTHON DIAGNOSTICS ----")

from fastapi import FastAPI as _FastAPI, Request # Alias to avoid conflict with modal.App and import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel as _BaseModel # Alias if needed

# Create a new FastAPI app instance for Modal
# This runs in a lightweight, serverless container (no GPU by default)
fastapi_app = _FastAPI(
    title="Clinical Note Summarizer (Modal)",
    description="A web application for generating summaries of clinical notes, deployed on Modal with GPU acceleration.",
    version="1.1.0-modal"
)

# Mount static files. The path must be correct *inside the container*.
fastapi_app.mount("/static", StaticFiles(directory=STATIC_FILES_PATH_IN_CONTAINER, check_dir=False), name="static")

class SummarizationRequest(_BaseModel): # Use aliased BaseModel
    text: str

@fastapi_app.get("/", response_class=HTMLResponse)
async def serve_frontend_modal(request: Request): # Use imported Request
    # Serve index.html from the path inside the container
    return FileResponse(INDEX_HTML_PATH_IN_CONTAINER)

@fastapi_app.post("/summarize")
async def summarize_modal_endpoint(payload: SummarizationRequest): # Changed 'request' to 'payload' for clarity
    input_text = payload.text or ""
    if not input_text.strip():
        return JSONResponse({"summary": "Input text is empty."})

    try:
        # Create a handle to our Modal class. Modal provisions the GPU container.
        model_runner = ClinicalT5Model()
        # Call the 'summarize' method. This call executes on the GPU container.
        summary_result = model_runner.summarize.remote(input_text)

        return JSONResponse({"summary": summary_result})

    except Exception as e:
        print(f"FastAPI Endpoint: Error calling ClinicalT5Model: {e}")
        return JSONResponse({"summary": f"Error processing request: {str(e)}"}, status_code=500)

# --- Expose FastAPI app via Modal's ASGI interface ---
@app.function(image=web_image)  # Use the dedicated web image
@modal.asgi_app() # Exposes the function as an ASGI app
def web_app():
    """
    This function returns the FastAPI app, allowing Modal to serve it.
    """
    return fastapi_app

# --- Optional: Local Entrypoint for Testing ---
# You can run this with: `modal run modal_deploy.py`
@app.local_entrypoint()
async def main_local_test():
    model = ClinicalT5Model() # Get a handle to the class
    test_note = "Patient complains of headache and nausea. Assessment: Possible migraine. Plan: Prescribe sumatriptan."
    print(f"\n[Local Test] Sending note to ClinicalT5Model: '{test_note}'")
    summary = model.summarize.remote(test_note) # Call the remote method
    print(f"[Local Test] Received summary: '{summary}'\n")