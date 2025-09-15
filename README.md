# CIFAR-10 MLOps Project 🚀

An end-to-end **MLOps pipeline** for image classification on the **CIFAR-10 dataset**, featuring model training, evaluation, inference, experiment tracking, containerization, and CI/CD.

---

## ✨ Features
- Train a **TinyVGG model** on CIFAR-10
- Evaluate model with confusion matrix & classification report
- Run inference on new images
- Serve predictions via a **FastAPI endpoint**
- Track experiments with **MLflow**
- Containerized with **Docker** for portability
- CI/CD pipeline with GitHub Actions

---

## 🛠 Tech Stack
- **Python 3.10**
- **PyTorch** (model + training)
- **scikit-learn** (evaluation metrics)
- **FastAPI** (serving API)
- **MLflow** (experiment tracking)
- **Docker** (containerization)
- **GitHub Actions** (CI/CD)

---

## 📂 Project Structure
'''
.
├── data/ # Dataset files
│ ├── cifar-10-batches-py/ # Extracted CIFAR-10 dataset
│ └── cifar-10-python.tar.gz # Original CIFAR-10 archive
├── docker-compose.yml # Docker Compose config (optional multi-service setup)
├── Dockerfile # Docker build file
├── full_image_name.txt # Helper file with Docker image name
├── mlruns/ # MLflow experiment tracking
│ ├── 0/ # Default experiment logs
│ └── models/ # MLflow model registry
├── models/ # Trained models and metadata
│ ├── confusion_matrix.png
│ ├── meta.json
│ └── tinyvgg_best.pth
├── requirements.txt # Python dependencies
├── src/ # Source code
│ ├── api.py # FastAPI app for serving predictions
│ ├── app.py # (Optional) app entry point
│ ├── data.py # Data loading utilities
│ ├── eval.py # Evaluation script
│ ├── inference.py # Inference script
│ ├── model.py # Model architecture (TinyVGG)
│ └── train.py # Training script
├── test_plane_image.jpg # Sample test image
├── README.md # Project documentation
└── structure.txt # Project structure dump
'''
---

## ⚙️ Setup

Clone the repo:
```bash
git clone https://github.com/your-username/cifar10-mlops.git
cd cifar10-mlops

Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

Install dependencies:
pip install -r requirements.txt

🚂 Training
python -m src.train

🔍 Evaluating the Model
python src/eval.py

🤖 Running Inference
python src/inference.py --image test_plane_image.jpg

API
python src/api.py

🐳 Docker Setup
Build the Docker image locally:
docker build -t cifar10-mlops .
Run the container:
docker run -p 8000:8000 cifar10-mlops

📊 MLflow Tracking
mlflow ui

⚙️ CI/CD Pipeline

GitHub Actions workflow (.github/workflows/ci-cd.yml) runs on:

push to main

pull_request to main

Pipeline Steps:

Checkout repo

Set up Python 3.10

Install dependencies from requirements.txt

Run unit tests (pytest tests/, optional)

Build Docker image from the current repo