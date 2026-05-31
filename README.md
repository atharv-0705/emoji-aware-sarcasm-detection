# Sarcoji — Sarcasm Detection

Detect sarcasm, sentiment, emotion, and bullying in social media text
using Hybrid Deep Learning (CNN-BiLSTM-Attention).

## Model Architecture
- **Embeddings**: GloVe-Twitter + Word2Vec + Emoji2Vec (600-dim fusion)
- **Feature Extraction**: Multi-scale CNN (kernels 2,3,4)
- **Sequence Modeling**: BiLSTM (64 units per direction)
- **Attention**: Multi-head Self-Attention (4 heads)
- **Training**: Stepwise Transfer Learning (3 phases)

## Performance
| Metric | Value |
|--------|-------|
| Accuracy | 84.5% |
| ROC-AUC | 93.0% |
| F1 Score | 81.5% |
| MCC | 0.670 |

## Deploy on HuggingFace Spaces

Click button below:

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR-USERNAME/sarcoji-app)

## Local Development

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
streamlit run app.py
```

Then visit: http://localhost:8501

## Files
- `app.py` — Combined Streamlit app (no FastAPI for HF Spaces)
- `artifacts/` — Model files (model.h5, vocab, threshold)
- `data/` — CSV analytics (metrics, stats, emoji analysis)
- `backend/` — FastAPI backend (for local development only)
- `frontend/` — Streamlit components (imported by app.py)

## Author
Atharv Gupta