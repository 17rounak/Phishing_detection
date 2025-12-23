import streamlit as st
import pandas as pd
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizerFast

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("bert_model_output")
tokenizer = BertTokenizerFast.from_pretrained("bert_model_output")

# Streamlit UI
st.set_page_config(page_title="Phishing Detector", page_icon="🛡️")
st.title("🛡️ Phishing Detection using BERT")

mode = st.radio("Choose input mode:", ["📱 SMS", "📧 Email", "📂 Upload CSV"])

def predict_message(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    logits = model(inputs).logits
    probs = tf.nn.softmax(logits, axis=1).numpy()[0]
    label = "Phishing 🚨" if tf.argmax(probs).numpy() == 1 else "Ham ✅"
    return label, probs.max()

if mode == "📱 SMS":
    sms_body = st.text_area("Enter SMS body:")
    if st.button("Predict SMS"):
        if sms_body.strip():
            label, confidence = predict_message(sms_body)
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
        else:
            st.warning("Please enter SMS text.")

elif mode == "📧 Email":
    email_subject = st.text_input("Enter Email Subject:")
    email_body = st.text_area("Enter Email Body:")
    if st.button("Predict Email"):
        if email_subject.strip() or email_body.strip():
            full_msg = f"{email_subject} {email_body}".strip()
            label, confidence = predict_message(full_msg)
            st.markdown(f"**Prediction:** {label}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
        else:
            st.warning("Please fill at least one of subject or body.")

elif mode == "📂 Upload CSV":
    file = st.file_uploader("Upload CSV file with `subject`, `body`, or `message` column", type=["csv"])
    if file:
        df = pd.read_csv(file)

        if 'message' in df.columns:
            df['message'] = df['message'].astype(str)
        elif 'body' in df.columns:
            df['body'] = df['body'].astype(str)
            df['subject'] = df.get('subject', '').astype(str)
            df['message'] = (df['subject'] + ' ' + df['body']).str.strip()
        else:
            st.error("CSV must have at least a 'body' or 'message' column.")

        st.write("📊 Preview of Uploaded Data:")
        st.dataframe(df.head())

        if st.button("Predict from File"):
            inputs = tokenizer(list(df['message']), return_tensors="tf", truncation=True,
                               padding=True, max_length=128)
            logits = model(inputs).logits
            probs = tf.nn.softmax(logits, axis=1).numpy()
            preds = tf.argmax(probs, axis=1).numpy()

            df['Prediction'] = ['Ham ✅' if p == 0 else 'Phishing 🚨' for p in preds]
            df['Confidence'] = probs.max(axis=1)

            st.success("✅ Prediction complete.")
            st.dataframe(df[['message', 'Prediction', 'Confidence']].head())

            # Downloadable result
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", data=csv, file_name='predictions.csv', mime='text/csv')
