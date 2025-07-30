import streamlit as st
import pandas as pd
import joblib
import email
from email import policy
from io import StringIO, BytesIO

@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("models/email_auth_model.pkl")

model = load_model()

st.title("📧 Email Authenticity Checker")

option = st.radio("Choose Input Method", ["🔤 Manual Entry", "📁 Upload CSV File", "📄 Upload .eml Email File"])

if option == "🔤 Manual Entry":
    domain = st.text_input("Sender's Domain (e.g. paypal.com)")
    subject = st.text_input("Subject")
    body = st.text_area("Email Body")

    if st.button("Check Authenticity"):
        text = f"{domain} {subject} {body}"
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0][pred]
        label = "🟢 Authorized" if pred == 1 else "🔴 Unauthorized"
        st.markdown(f"### Prediction: {label}")
        st.write(f"Confidence: {prob:.2%}")

elif option == "📁 Upload CSV File":
    st.markdown("Upload a CSV file with the following columns: **domain, subject, body**")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if {"domain", "subject", "body"}.issubset(df.columns):
                df["text"] = df["domain"] + " " + df["subject"] + " " + df["body"]
                df["Prediction"] = model.predict(df["text"])
                df["Confidence"] = model.predict_proba(df["text"]).max(axis=1)
                df["Prediction Label"] = df["Prediction"].map({1: "🟢 Authorized", 0: "🔴 Unauthorized"})
                st.dataframe(df[["domain", "subject", "body", "Prediction Label", "Confidence"]])
            else:
                st.error("CSV must contain columns: domain, subject, body")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

elif option == "📄 Upload .eml Email File":
    st.markdown("Upload a **.eml** email file")
    eml_file = st.file_uploader("Choose a .eml file", type=["eml"])

    if eml_file:
        try:
            msg = email.message_from_binary_file(eml_file, policy=policy.default)
            subject = msg["subject"] or ""
            from_header = msg["from"] or ""
            domain = from_header.split("@")[-1] if "@" in from_header else from_header
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_content().strip()
                        break
            else:
                body = msg.get_content().strip()

            st.write("**Extracted Email Content:**")
            st.write(f"📧 **From Domain:** `{domain}`")
            st.write(f"📝 **Subject:** `{subject}`")
            st.write(f"📄 **Body Preview:** `{body[:300]}...`" if len(body) > 300 else body)

            text = f"{domain} {subject} {body}"
            pred = model.predict([text])[0]
            prob = model.predict_proba([text])[0][pred]
            label = "🟢 Authorized" if pred == 1 else "🔴 Unauthorized"
            st.markdown(f"### Prediction: {label}")
            st.write(f"Confidence: {prob:.2%}")

        except Exception as e:
            st.error(f"Failed to parse .eml file: {e}")
