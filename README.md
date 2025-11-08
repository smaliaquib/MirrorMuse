# MirrorMuse - End-to-End RAG System  

## Overview  

MirrorMuse is a production-ready LLM framework covering the full development lifecycle:

- ** Data Pipeline**: Web crawling, dataset creation, and preprocessing
- ** Model Training**: Fine-tuning (SFT & DPO) with evaluation
- ** RAG System**: Retrieval-Augmented Generation implementation
- ** Cloud Deployment**: AWS SageMaker integration
- ** Monitoring**: Comet ML + Opik for tracking

**Pre-trained models available on [Hugging Face](https://huggingface.co/SkillRipper)**

---

## 🔧 Installation  

### Prerequisites  
- Python 3.11+ (`pyenv install 3.11.8`)
- [UV](https://github.com/astral-sh/uv) (Fast Python package installer)
- Docker 27.1.1+
- AWS CLI (for cloud deployment)

### Quick Setup  
```bash
git clone https://github.com/smaliaquib/MirrorMuse.git
cd MirrorMuse

# Install with UV
uv pip install --pyproject .

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Required Environment Variables
```
OPENAI_API_KEY=your_key
HUGGINGFACE_ACCESS_TOKEN=your_token
COMET_API_KEY=your_key
```

---

## Infrastructure  

### Local Services  
| Service          | Connection String                                 |
|------------------|---------------------------------------------------|
| MongoDB          | `mongodb://mirrormuse:mirrormuse@127.0.0.1:27017` |
| Qdrant           | `http://localhost:6333`                           |
| ZenML Dashboard  | `http://localhost:8237`                           |

### Cloud Services  
- AWS SageMaker (Training/Inference)
- Qdrant Cloud (Vector DB)
- MongoDB Atlas (Document Store)

---

## Quick Start  

### Data Pipeline  
```bash
# Run ETL
uv run tools/run.py --run-etl --no-cache --etl-config-filename digital_data_etl.yml

# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Feature Engineering  
uv run tools/run.py --no-cache --run-feature-engineering

# Export data
uv run tools/data_warehouse.py --export-raw-data
```

### Model Training  
```bash
uv run tools/run.py --run-training --config training_config.yml
```

### Deployment  
```bash
uv run tools/deploy.py --env prod
```

---

## Usage Notes  

1. All pipelines support `--no-cache` for fresh runs
2. Check `configs/` for pipeline configurations
3. Monitor runs in Comet ML/Opik dashboards
4. Production deployments require AWS credentials

---

## Troubleshooting  

- **Qdrant issues**: Ensure ports 6333-6334 are free
- **UV installation**: Use `pip install --force-reinstall uv`
- **Missing dependencies**: Run `uv pip install --pyproject . --reinstall`
---
