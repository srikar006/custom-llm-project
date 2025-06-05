# Custom LLM Project (T5-base)

This project shows how to fine-tune a medium-sized T5 transformer using Hugging Face on a simple Q&A dataset.

## 📁 Structure
- `sample_data.csv` – CSV format training data
- `train.py` – Training script using Hugging Face `Trainer`
- `Dockerfile` – Build and run your model anywhere
- `requirements.txt` – Python package dependencies

## 🚀 Train in Docker

```bash
docker build -t custom-llm .
docker run --rm custom-llm
