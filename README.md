# Phishing_detection
BERT-based phishing detection system for SMS and emails with a Streamlit web interface. Supports single message prediction and bulk CSV analysis, with optional Colab + ngrok deployment for live demos.

This project implements a BERT-based phishing detection system that classifies SMS and email messages as either Phishing or Ham (legitimate). The system uses a fine-tuned BERT (bert-base-uncased) model and provides an interactive web interface built with Streamlit. It supports three modes of input: SMS text, email input using subject and body, and bulk prediction through CSV file upload. The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, and deployment-oriented inference.

During training, email subject and body are combined into a single message, missing values are handled, and labels are normalized such that all legitimate messages are mapped to class 0 and all phishing or malicious messages are mapped to class 1. Text is tokenized using the BERT tokenizer and the model is fine-tuned using TensorFlow. Model performance is evaluated using accuracy, ROC-AUC score, confusion matrix, and loss and accuracy curves across epochs. The trained model and tokenizer are saved locally after training. Due to size and privacy considerations, datasets and trained model artifacts are intentionally not included in this repository.

To run the project locally, install dependencies using `pip install -r requirements.txt`. If retraining is required, place the dataset at `data/Book1.csv` (expected columns: subject, body, label) and run `python train_bert_phishing.py`. To start the application, run `streamlit run app.py`, after which the interface will be available at `http://localhost:8501`.

When running the application on Google Colab, direct access to localhost is not possible. In this case, ngrok is used to expose the Streamlit app publicly. First install the required tools using `pip install streamlit pyngrok`. Add your ngrok authentication token using `ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN`. Start the Streamlit app in the background using `streamlit run app.py &>/content/log.txt &`. After waiting a few seconds, expose the application by running:

from pyngrok import ngrok
ngrok.kill()
public_url = ngrok.connect(8501)
print("Your public Streamlit app URL:", public_url)

Opening the printed URL will launch the live application. Once the demo is complete, it is important to stop all running services to free system resources. This can be done using the commands `pkill streamlit` and `pkill ngrok`, which terminate the Streamlit server and the ngrok tunnel.

The Streamlit interface allows users to input SMS text, email subject and body, or upload a CSV file for batch prediction. For CSV uploads, the file must contain either a `message` column, a `body` column, or both `subject` and `body`. The application automatically preprocesses the data, generates predictions, displays confidence scores, and allows the results to be downloaded as a CSV file.

This repository contains the Streamlit application code, the BERT training and evaluation script, dependency specifications, and configuration files. Model artifacts and datasets are excluded using `.gitignore` to maintain repository cleanliness and follow best practices. The project demonstrates practical skills in natural language processing, transformer-based modeling, machine learning evaluation, and web-based deployment of AI systems.

Author: Rounak Handa
