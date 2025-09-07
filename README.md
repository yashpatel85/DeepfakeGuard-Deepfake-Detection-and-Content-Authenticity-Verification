# DeepfakeGuard 🔍  
A deep learning-based system for detecting deepfake images (AI-generated faces) using **Xception CNN architecture**.  

## 📌 Overview
Deepfakes pose a growing threat to digital media integrity. This project builds a **deepfake detection model** that can classify images as **REAL** or **FAKE** with high accuracy.  

The system includes:
- **Data preprocessing & augmentation**
- **Xception-based CNN training**
- **Evaluation with metrics (accuracy, precision, recall, F1, confusion matrix)**
- **FastAPI inference server**
- (Optional) **Docker + AWS deployment** (in progress)

---

## 📂 Project Structure
DeepfakeGuard/
│── data_faces_split/ # Dataset (train/val/test splits)
│── src/
│ ├── train_efficient.py # Training script (Xception/EfficientNet)
│ ├── inference.py # Inference script for single image
│ ├── main.py # FastAPI app for deployment
│── models/
│ └── best_model.pth # Saved trained model
│── requirements.txt # Dependencies
│── Dockerfile # Containerization setup
│── README.md # Project documentation

## 🚀 Installation
### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/DeepfakeGuard.git
cd DeepfakeGuard

2️⃣ Create Environment
conda create -n deepfakeguard python=3.10 -y
conda activate deepfakeguard

3️⃣ Install Dependencies
pip install -r requirements.txt

🏋️‍♂️ Training
python src/train_xception.py

🔎 Inference
python src/inference.py data_faces_split/test/fake/000_003_2.jpg


Example Output:

{
  "real": 0.52,
  "fake": 0.48,
  "decision_threshold": 0.42,
  "predicted_label": "FAKE"
}


📊 Results
Classification Report:
              precision    recall  f1-score   support

        fake       1.00      0.96      0.98      1500
        real       0.96      1.00      0.98      1500

    accuracy                           0.98      3000
   macro avg       0.98      0.98      0.98      3000
weighted avg       0.98      0.98      0.98      3000


Confusion Matrix:
[[1433   67]
 [   6 1494]]


✅ Accuracy: 98%
✅ Balanced detection of REAL and FAKE

Docker (Optional)

Build and run container:

docker build -t deepfakeguard .
docker run -p 8000:8000 deepfakeguard

☁️ AWS Deployment (Future Scope)

Deploy Docker image to AWS ECR

Run on AWS EC2 (GPU instance)

Auto-scale inference service

🛠️ Tech Stack

Python 3.10

PyTorch

timm (Xception)

FastAPI + Uvicorn

scikit-learn (metrics)

Docker + AWS (deployment)

📜 License

MIT License – free to use and modify.



## Large Data / Models

Due to GitHub file size limits, large files are hosted externally:

- [Processed Datafiles](https://drive.google.com/drive/folders/153ofbLzdjIwz_GOrbozm9x9pz_oWc20y?usp=drive_link) (~5.66 GB)


After downloading, extract the files into the appropriate directories as mentioned in the project instructions.
