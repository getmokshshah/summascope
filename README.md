# SummaScope — AI-Powered Document Analyzer

An end-to-end NLP pipeline that extracts abstractive summaries, named entities, and key phrases from any document. Built with pre-trained transformer models and deployed as an interactive web demo via Gradio on Hugging Face Spaces.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

SummaScope combines multiple NLP capabilities into a single, cohesive analysis tool. Given any English text — a news article, research paper, blog post, or report — it produces:

- **Abstractive Summary** — a concise rewrite (not just extraction) of the key points, powered by DistilBART
- **Named Entities** — people, organizations, locations, and other proper nouns identified by BERT-NER
- **Key Phrases** — the most salient topics and concepts via YAKE keyword extraction
- **Text Statistics** — word count, sentence count, and estimated reading time

The project demonstrates practical ML engineering: model selection, inference optimization, chunking strategies for long documents, and deployment on resource-constrained infrastructure (Hugging Face Spaces free tier, CPU-only).

## Architecture

```
SummaScope Pipeline
│
├─ Input Processing
│   ├── Whitespace normalization & cleaning
│   └── Chunking for long documents (>900 tokens)
│
├─ Abstractive Summarization
│   ├── Model: sshleifer/distilbart-cnn-12-6  (~1.2 GB)
│   ├── Strategy: chunk → summarize → concatenate
│   └── Configurable target length (10–50% of original)
│
├─ Named Entity Recognition
│   ├── Model: dslim/bert-base-NER  (~400 MB)
│   ├── Labels: Person, Organization, Location, Miscellaneous
│   └── Deduplication + confidence filtering (>0.75)
│
├─ Key Phrase Extraction
│   ├── Algorithm: YAKE (Yet Another Keyword Extractor)
│   ├── Unsupervised, no model required
│   └── Bi-gram extraction with deduplication
│
└─ Text Statistics
    ├── Word & sentence count
    └── Estimated reading time (238 wpm average)
```

### Model Selection Rationale

| Component | Model | Why This Model |
|---|---|---|
| Summarization | `distilbart-cnn-12-6` | Distilled from BART-large-CNN. Retains ~95% of BART's summarization quality at ~60% of the size. Runs on CPU in <10s for typical articles. |
| NER | `dslim/bert-base-NER` | Fine-tuned on CoNLL-2003. Strong F1 (~91%) with a small footprint (~400 MB). Fast inference on CPU. |
| Key Phrases | YAKE | Unsupervised — no model to load, zero additional memory. Language-agnostic and fast. Produces high-quality bi-gram keyphrases. |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/getmokshshah/summascope.git
cd summascope
pip install -r requirements.txt
```

### 2. Launch the Demo

```bash
python app.py
```

The Gradio interface launches at `http://localhost:7860`.

### 3. Use via API

SummaScope exposes a `/predict` endpoint via Gradio's API:

```python
from gradio_client import Client

client = Client("getmokshshah/summascope")
summary, entities, phrases, stats = client.predict(
    "Your document text here...",
    30,  # summary length percentage
    api_name="/predict",
)
```

## Deployment

### Hugging Face Spaces

1. Create a new Space (Gradio SDK, CPU Basic)
2. Upload `app.py`, `requirements.txt`, and this `README.md`
3. The Space auto-builds and deploys

The app runs on the free CPU tier (2 vCPU, 16 GB RAM). Model loading takes ~30s on cold start; inference is typically 3–8s per document.

### Custom Website Integration

SummaScope can be embedded in any website using the [Gradio JS Client](https://www.gradio.app/docs/js-client):

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("getmokshshah/summascope");
const result = await client.predict("/predict", [text, summaryLength]);
// result.data = [summary, entitiesJson, keyPhrases, stats]
```

## Project Structure

```
summascope/
├── app.py              # Gradio interface + NLP pipeline
├── requirements.txt    # Python dependencies
└── README.md
```

## Example Output

**Input** (210 words about the Transformer architecture):

| Output | Result |
|---|---|
| **Summary** | The Transformer architecture, introduced in 2017, changed NLP by using self-attention to process sequences in parallel rather than step by step. Its multi-head attention mechanism allows models to attend to different positions simultaneously. Variants like BERT and GPT use only the encoder or decoder. The architecture now underpins virtually all state-of-the-art language models and has expanded into vision, protein prediction, and audio. |
| **Entities** | Google (Organization), Vaswani (Person) |
| **Key Phrases** | Transformer architecture, self-attention, multi-head attention, language models, natural language processing |
| **Stats** | 210 words · 12 sentences · ~1 min read |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) for the model hub and inference pipeline
- [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) for the summarization model
- [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) for the NER model
- [YAKE](https://github.com/LIAAD/yake) for unsupervised keyword extraction
- [Gradio](https://gradio.app) for the web interface
