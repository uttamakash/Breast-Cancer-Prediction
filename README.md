🩺 Breast Cancer Prediction using Logistic Regression
📘 Overview

This project is a machine learning-based web application that predicts whether a breast tumor is malignant (cancerous) or benign (non-cancerous) based on input features such as mean radius, mean texture, mean perimeter, mean area, and mean smoothness.

The model uses Logistic Regression from the scikit-learn library and is deployed using a Flask web framework.

🚀 Features

🧠 Machine Learning Model (Logistic Regression)

🌐 Web Interface built with Flask and HTML/CSS

📊 Automatic data scaling using StandardScaler

⚙️ Real-time prediction with model confidence score

🧾 Well-structured and easy-to-understand codebase

📂 Project Structure
cancer-predictor/
│
├── static/
│   └── style.css              # Frontend styling
├── templates/
│   ├── index.html             # Input form page
│   └── result.html            # Prediction result page
│
├── model.pkl                  # Trained Logistic Regression model
├── scaler.pkl                 # Scaler used for feature normalization
├── train_model.py             # Python script to train and save the model
├── app.py                     # Flask application file
├── requirements.txt           # Required dependencies
└── README.md                  # Project documentation

⚙️ Installation
🪄 1. Clone the repository
git clone https://github.com/<your-username>/cancer-predictor.git
cd cancer-predictor

🧰 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # for Windows
# OR
source venv/bin/activate   # for macOS/Linux

📦 3. Install dependencies
pip install -r requirements.txt


(If requirements.txt is not present, install manually:)

pip install flask scikit-learn pandas numpy

🧮 4. Train the Model
python train_model.py


This will generate:

model.pkl

scaler.pkl

🌍 5. Run the Flask App
python app.py


The app will start at:
👉 http://127.0.0.1:5000

💻 Usage

Enter the required tumor features in the web form.

Click Predict.

View the result:

🔴 Malignant (Positive) — Cancer detected.

🟢 Benign (Negative) — No cancer detected.

The model also shows confidence percentage and a motivational message.

📈 Model Details

Algorithm Used: Logistic Regression

Libraries: scikit-learn, pandas, numpy

Accuracy: ~97% on test data

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (from scikit-learn)

🧠 Key Learnings

End-to-end ML pipeline: data preprocessing, model training, and deployment

Building interactive Flask web apps

Feature scaling and model persistence using pickle

User interface design for ML prediction systems

🧩 Future Improvements

Add more features for better accuracy

Deploy on cloud platforms (Heroku, AWS, or Render)

Include user authentication

Create visual graphs for predictions

👨‍💻 Author

Uttam Akash

<img width="1919" height="918" alt="Screenshot 2025-11-07 014538" src="https://github.com/user-attachments/assets/8557ef5d-6cfb-4b2f-9d3a-1602ee67c57d" />
