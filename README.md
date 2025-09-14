
# 🧠 MOR-GPT

**MOR-GPT** is a from-scratch implementation of a transformer-based **Large Language Model (LLM)** designed for domain-specific conversational intelligence.  
It integrates a complete MLOps pipeline for training, versioning, deployment, and monitoring — bridging cutting-edge research and production engineering.

---

## 📜 Abstract

MOR-GPT explores how small, domain-adapted LLMs can be constructed from first principles without relying on pre-trained checkpoints.  
The model was built to understand and respond to technical queries related to AI, ML, and MLOps, and serves as a foundation for future scalable LLM architectures.

---

## 🎯 Objectives

- Build a complete LLM stack from scratch: tokenizer → transformer → trainer → inference.
- Deploy the model in a production-grade MLOps pipeline.
- Enable real-time chat-based interaction through a web interface.
- Provide full experiment reproducibility and version control.

---

## 🧩 System Architecture

        ┌───────────────┐
        │   Raw Corpus  │
        └──────┬────────┘
               │
        ┌──────▼────────┐
        │ Custom Tokenizer│
        └──────┬────────┘
               │
    ┌──────────▼────────────┐
    │ Transformer Architecture │
    │ (Encoder-Decoder)       │
    └──────────┬────────────┘
               │
        ┌──────▼────────┐
        │ Training Engine │
        │ (PyTorch)        │
        └──────┬────────┘
               │
   ┌───────────▼────────────┐
   │ FastAPI Inference API  │
   └──────┬────────────────┘
          │
   ┌──────▼─────────┐
   │ React Frontend │
   └────────────────┘

---

## ⚙️ Core Components

### 📝 Tokenizer
- Built from a domain-specific dataset (AI/ML/MLOps concepts, personal Q&A)
- Byte Pair Encoding (BPE)-like tokenization
- `<pad>, <unk>, <bos>, <eos>` special tokens
- Vocabulary size: 3000

### 🧠 Model Architecture
- Transformer-based (Encoder-Decoder)
- Multi-head self-attention
- Feed-forward layers with residual connections and layer norm
- Positional encodings and attention masking
- Teacher forcing during training

### 📊 Training Pipeline
- Data preprocessing and batching
- Cross-entropy loss with gradient clipping
- Checkpointing and resume support
- Metrics tracked via **MLflow**

### 🖥️ Deployment & MLOps
- Inference served via **FastAPI**
- Containerized with **Docker**
- CI/CD automation using **GitHub Actions**
- Experiment tracking and model registry using **MLflow**

### 💬 Frontend Interface
- Built with **React** + **TypeScript**
- Real-time chat interface
- Model selection and progress indicators
- Displays training metrics and response logs

---

## 📁 Repository Structure


---

## ⚡ Tech Stack

| Layer        | Technology           |
|--------------|--------------------------|
| Model        | PyTorch                  |
| API Backend  | FastAPI                  |
| Frontend     | React + TypeScript       |
| MLOps         | MLflow, Docker, GitHub Actions |
| Language     | Python, TypeScript       |

---

## 📈 Current Status

- ✅ Tokenizer implemented  
- ✅ Transformer architecture implemented  
- ✅ Data pipeline complete  
- ✅ Backend and Frontend fully integrated  
- ⚡ Training run in progress  
- 📌 Evaluation benchmarks coming soon  

---

## 📊 Example Outputs *(Coming Soon)*

- Training loss and accuracy plots
- Sample inference responses
- MLflow experiment dashboard screenshots
- Tokenization samples

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/RayAKaan/MOR-GPT_Model.git
cd MOR-GPT_Model

### Backend setup
cd app
uvicorn main:app --reload

### Frontend setup
cd frontend
npm install
npm run dev
