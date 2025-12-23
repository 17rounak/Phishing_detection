# Project Workflow and Design

This document explains the end-to-end workflow of the phishing detection system, from data ingestion and model training to deployment and inference.

The goal of the project is to detect phishing attempts in SMS and email messages using a transformer-based language model. The system is designed to be modular, reproducible, and deployment-friendly.

The workflow begins with data preparation. Raw email and SMS data is provided in CSV format. For email data, the subject and body are combined into a single message string. Missing values are handled gracefully, and messages with no textual content are discarded. Labels are normalized so that all legitimate messages are mapped to class 0 (ham) and all phishing or malicious messages are mapped to class 1.

During the training phase, text data is tokenized using the BERT tokenizer corresponding to the `bert-base-uncased` model. The tokenized inputs are then fed into a TensorFlow-based BERT sequence classification model. The model is fine-tuned using supervised learning with cross-entropy loss. Early stopping is used to prevent overfitting, and the best-performing model is selected based on validation accuracy. After training, the model and tokenizer are saved locally for later inference.

Evaluation is performed using multiple metrics to ensure robust performance. These include overall accuracy, ROC-AUC score, confusion matrix, and loss/accuracy curves across epochs. These metrics provide insight into both classification performance and generalization ability.

For inference and deployment, the trained model is loaded into a Streamlit application. The web interface allows users to interact with the model in three different ways: SMS input, email input with subject and body, and bulk prediction via CSV file upload. For each input, the same preprocessing and tokenization steps used during training are applied to ensure consistency. The model outputs a probability score, which is converted into a phishing or ham prediction along with a confidence value.

Because Streamlit applications running on Google Colab cannot be accessed directly through localhost, ngrok is used to expose the application to the public internet during demos. The Streamlit server is started in the background, and ngrok creates a secure tunnel to port 8501. A public URL is generated, which can be shared and accessed through a browser. Once the demo is complete, both Streamlit and ngrok processes are terminated to free system resources.

Throughout the project, best practices are followed to ensure security and maintainability. Raw datasets and trained model artifacts are excluded from the GitHub repository to prevent privacy issues and repository bloat. Configuration files such as `.gitignore` and `requirements.txt` are used to keep the project clean and reproducible.

This workflow demonstrates a complete machine learning lifecycle, combining natural language processing, transformer-based modeling, evaluation, and real-world deployment considerations.
