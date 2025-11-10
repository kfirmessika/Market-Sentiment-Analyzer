# Market Sentiment Analyzer

## Overview
Market Sentiment Analyzer is a research-driven collection of Jupyter notebooks for exploring and fine-tuning transformer models on financial microblog data. The workflow walks through exploratory data analysis, systematic hyper-parameter search, targeted fine-tuning of multiple architectures, and real-time inference against live social feeds. The project centers on the Hugging Face synthetic financial tweets sentiment dataset and produces ready-to-evaluate checkpoints for production-style experimentation.

## Dataset
All experiments rely on the **TimKoornstra/synthetic-financial-tweets-sentiment** dataset hosted on the Hugging Face Hub. Each notebook loads the parquet shard directly via the Hub file system interface (e.g., `hf://datasets/TimKoornstra/synthetic-financial-tweets-sentiment/data/train-00000-of-00001.parquet`). The corpus contains short-form tweets labeled with `0 = Neutral`, `1 = Bullish`, and `2 = Bearish` sentiment classes, enabling balanced sampling strategies throughout the notebooks.

## Repository Structure
The project is organized into sequential notebooks that can be executed independently or as part of a broader experimentation pipeline:

| Notebook | Purpose |
| --- | --- |
| `Part1_Data_Analysis_For_Fine_Tuning.ipynb` | Exploratory data analysis (EDA) of the full dataset, including distribution plots for sentiment classes and tweet lengths. |
| `Part2_Grid_Search_Final_Project.ipynb` | Custom hyper-parameter search across FinBERT, BERT Tiny, BERTweet, and Twitter RoBERTa with configurable optimizers, learning rates, epochs, and seeds. |
| `Part3_Bert_Tiny_Fine_Tune.ipynb` | Fine-tunes `prajjwal1/bert-tiny` with Hugging Face `Trainer`, evaluates with accuracy/F1 metrics, and inspects performance via confusion matrices. |
| `Part4_Distilbert_Fine_Tune.ipynb` | Continues experimentation with a DistilBERT-based sentiment model, balancing the dataset, training with AdamW, and saving checkpoints to persistent storage. |
| `Part5_Roberta_Freezing_Fine-Tune.ipynb` | Freezes the base layers of `cardiffnlp/twitter-roberta-base-sentiment-latest` to train only the classification head using Hugging Face `Trainer`. |
| `Part6_Roberta_Fine_Tune.ipynb` | Runs a full fine-tuning cycle of the same RoBERTa architecture across multiple balanced dataset splits with manual PyTorch training loops. |
| `Part7_Real_Time_Tweets.ipynb` | Fetches live StockTwits messages, applies both base and fine-tuned versions of the supported models, and exports aggregated predictions for real-time analysis. |

## Getting Started

### Prerequisites
* Python 3.9+
* CUDA-capable GPU (recommended for training notebooks)
* Access to the Hugging Face Hub (datasets and pretrained model checkpoints)

### Recommended Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust for your CUDA version
pip install transformers datasets accelerate evaluate scikit-learn pandas matplotlib tqdm safetensors requests
```

> **Note:** Some notebooks use Google Colab utilities such as `drive.mount`. When running locally, replace these sections with equivalent local filesystem paths.

### Downloading the Dataset Locally (Optional)
If you prefer to cache the dataset instead of streaming from the Hub, use the `huggingface_hub` or `datasets` library to download the parquet shard:
```python
from datasets import load_dataset

dataset = load_dataset("TimKoornstra/synthetic-financial-tweets-sentiment")
dataset["train"].to_pandas().to_parquet("financial-tweets.parquet")
```
Update the notebook paths accordingly (e.g., `pd.read_parquet("financial-tweets.parquet")`).

## Running the Notebooks
Each notebook is self-contained. A typical workflow is:

1. **EDA (Part 1)** – Inspect label balance, tweet length distribution, and general dataset health.
2. **Hyper-parameter Search (Part 2)** – Run the grid-search notebook to benchmark multiple architectures and gather baseline accuracy metrics.
3. **Model-Specific Fine-Tuning (Parts 3–6)** – Choose the architecture that best fits your requirements and execute the corresponding notebook. Save checkpoints locally or to cloud storage for later inference.
4. **Real-Time Scoring (Part 7)** – Configure model paths in the notebook (pointing to saved `.pt` or `.safetensors` weights) and generate sentiment predictions for live StockTwits streams.

Because the notebooks use moderate-to-large transformer checkpoints, ensure that your runtime has sufficient GPU memory or adjust batch sizes and sequence lengths as needed. For reproducibility, set the random seeds provided in each notebook before training.

## Real-Time Inference Pipeline
The `Part7_Real_Time_Tweets.ipynb` notebook orchestrates an end-to-end sentiment pipeline:

1. Fetches the latest messages for a target stock symbol via the StockTwits public API.
2. Loads untuned and fine-tuned versions of each supported model while sharing the tokenizer with the original checkpoint.
3. Performs batched predictions, mapping raw logits back to human-readable sentiment labels.
4. Consolidates predictions into a CSV file for downstream analysis or dashboarding.

You can adapt the notebook into a standalone Python script by moving the helper functions (`load_model_and_tokenizer`, `predict_sentiment`, `fetch_stocktwits`, and `main`) into a module and wiring them into your existing data pipelines.

## Experiment Tracking & Evaluation
* **Metrics:** Accuracy, precision, recall, and F1 scores are calculated for fine-tuned models using Hugging Face’s `Trainer` callbacks or manual evaluation loops.
* **Confusion Matrices:** Visualizations in Part 3 help diagnose class-level performance and highlight areas for further data augmentation or rebalancing.
* **Checkpoint Management:** Parts 4–6 show how to persist models in Google Drive or local directories to enable continuous experimentation without retraining from scratch.

## Extending the Project
* Introduce experiment tracking with tools such as Weights & Biases or MLflow to log training runs across the notebooks.
* Package the real-time inference helpers into a FastAPI service for production deployment.
* Explore additional transformer backbones (e.g., DeBERTa, Longformer) or lightweight distilled models for latency-sensitive applications.
* Incorporate risk-aware analytics by correlating sentiment trends with market indicators or portfolio performance.

## Acknowledgements
* Dataset courtesy of [TimKoornstra](https://huggingface.co/TimKoornstra) via the Hugging Face Hub.
* Pretrained transformer checkpoints from Hugging Face model authors cited in each notebook.

## License
This repository does not currently include a license. If you plan to distribute or commercialize derivative work, please add an appropriate LICENSE file and verify the usage terms of the datasets and pretrained models referenced above.
