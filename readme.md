# SummLLM

## Core Aim

Design and implement an LLM-based model for text summarization. This project requires reviewing the literature and source code of LLM-based summarization methods, selecting a base model, and developing a prototype system that accepts text input and produces summaries.

## Prerequisites

- Windows (PowerShell)
- Conda (Anaconda or Miniconda)
  - https://docs.conda.io/en/latest/miniconda.html

## Quick Start

1. Install Conda and initialize PowerShell integration:

   ```powershell
   conda init powershell
   # Close and reopen PowerShell or restart the VS Code terminal
   ```

2. Create and activate a Python virtual environment:

   ```powershell
   conda create -n summllm python=3.10 -y
   conda activate summllm
   ```

3. Install Python dependencies (run from project root):

   ```powershell
   pip install -r requirements.txt
   ```

   - For GPU support, follow PyTorch installation instructions at https://pytorch.org/

4. Prepare model weights

   - A downloader script is provided at backend/models/download_models.py.
   - Default model: facebook/bart-large-cnn
   - Example:
     ```powershell
     python .\backend\models\download_models.py
     # Force re-download:
     python .\backend\models\download_models.py -f -o .\backend\models\models--facebook--bart-large-cnn
     ```
   - The repository expects models under `backend/models/`.

   - The downloader script:

     ```python
     import argparse
     import logging
     from pathlib import Path

     try:
         from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
     except Exception as e:
         raise RuntimeError("Please install transformers and torch. Error: " + str(e))

     logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

     def download_and_save(model_name: str, out_dir: str, force: bool = False):
         out_path = Path(out_dir).expanduser().resolve()
         if out_path.exists() and any(out_path.iterdir()) and not force:
             logging.info("Target directory exists and is not empty. Skipping download: %s", out_path)
             return

         out_path.mkdir(parents=True, exist_ok=True)
         logging.info("Downloading model: %s -> %s", model_name, out_path)

         logging.info("Downloading tokenizer...")
         tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
         tokenizer.save_pretrained(out_path)

         logging.info("Downloading model weights...")
         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
         model.save_pretrained(out_path)

         logging.info("Download complete: %s", out_path)

     def main():
         default_out = Path(__file__).parent / "models--facebook--bart-large-cnn"
         parser = argparse.ArgumentParser(description="Download and save a HuggingFace model locally")
         parser.add_argument("-m", "--model", default="facebook/bart-large-cnn", help="HuggingFace model name")
         parser.add_argument("-o", "--out", default=str(default_out), help="Output directory")
         parser.add_argument("-f", "--force", action="store_true", help="Force re-download if target exists and is not empty")
         args = parser.parse_args()

         download_and_save(args.model, args.out, args.force)

     if __name__ == "__main__":
         main()
     ```

5. Environment variables

   - (PowerShell):
     ```powershell
     $env:LOCAL_MODEL_PATH="Your-model-path\models--facebook--bart-large-cnn\snapshots\<snapshot-id>"
     ```

6. Start backend (FastAPI)

   ```powershell
   uvicorn backend.main:app --reload
   ```

   - API docs: http://localhost:8000/docs

7. Start frontend (Streamlit)
   ```powershell
   streamlit run frontend/app.py
   ```
