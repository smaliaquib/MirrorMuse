# MirrorMuse End-to-End LLM System  

This repository contains the code and resources for building a production-ready LLM-based system, covering everything from data collection to deployment and monitoring.  

## 📌 Overview  

The project guides you through creating a complete LLM pipeline:  

- **📝 Data Collection & Generation**: Web crawling, dataset creation, and preprocessing.  
- **🔄 LLM Training Pipeline**: Fine-tuning (SFT & DPO) and evaluation.  
- **📊 Simple RAG System**: Retrieval-Augmented Generation for enhanced responses.  
- **🚀 AWS Deployment**: Scalable cloud deployment using SageMaker.  
- **🔍 Monitoring**: Experiment tracking (Comet ML) and prompt monitoring (Opik).  
- **🧪 Testing & Evaluation**: CI/CD and model performance checks.  

The final trained model is available on **Hugging Face**.  

---

## 🔧 Installation  

### Prerequisites  
- **Python 3.11** (Recommended: `pyenv install 3.11.8`)  
- **Poetry** (>= 1.8.3) for dependency management  
- **Docker** (>= 27.1.1) for local infrastructure  
- **AWS CLI** (for cloud deployment)  

### Setup  
1. **Clone the repository**:  
   ```bash
   git clone https://github.com/smaliaquib/MirrorMuse.git
   cd MirrorMuse
   ```

2. **Install dependencies**:  
   ```bash
   uv init
   uv pip install --pyproject .   
   ```

3. **Set up environment variables**:  
   Copy `.env.example` to `.env` and fill in your API keys:  
   ```bash
   cp .env.example .env
   ```
   Required keys:  
   - `OPENAI_API_KEY`  
   - `HUGGINGFACE_ACCESS_TOKEN`  
   - `COMET_API_KEY` (for Comet ML & Opik)  
   ```

---

## 🏗️ Infrastructure  

### Local Development  
- **MongoDB**: `mongodb://mirrormuse:mirrormuse@127.0.0.1:27017`  
- **Qdrant**: `http://localhost:6333`  
- **ZenML Dashboard**: `http://localhost:8237`  

### Cloud Deployment  
- **AWS SageMaker**: For model training & inference  
- **Qdrant Cloud**: Vector database  
- **MongoDB Atlas**: NoSQL database  

---

Here's a concise `README.md` section that includes these commands for easy reference:

## 🚀 Quick Start Commands

### 1. Run Data ETL Pipeline
```bash
uv run tools/run.py --run-etl --no-cache --etl-config-filename digital_data_etl.yml
```

### 2. Start Qdrant Vector Database (Docker)
```bash
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 3. Run Feature Engineering Pipeline
```bash
uv run tools/run.py --no-cache --run-feature-engineering
```

### 4. Export Raw Data from Warehouse
```bash
uv run tools/data_warehouse.py --export-raw-data
```

## 📝 Usage Notes
1. For the ETL pipeline, make sure you have a valid `digital_data_etl.yml` configuration file
2. The Qdrant container runs in detached mode (`-d`) with ports 6333 (HTTP) and 6334 (gRPC) exposed
3. Use `--no-cache` flag to force fresh execution of pipelines
4. Exported raw data will be saved in JSON format by default

## 🔧 Prerequisites
- Python 3.11+ with UV installed (`pip install uv`)
- Docker installed and running
- Valid configuration files in place
- Required environment variables set in `.env` file

For the complete setup instructions, please refer to the main [Installation](#-installation) section.
