import streamlit as st
import numpy as np
import pandas as pd
import os
import pdfplumber
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from groq import Groq
from paddleocr import PaddleOCR
from pdf2image import convert_from_bytes

# Load environment variables
load_dotenv()

# GROQ client setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# OCR reader supporting English and Traditional Chinese
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# Benefit code dictionary (example)
benefit_codes = {
    "X1001": "General Consultation",
    "X2002": "Lab Test",
    "X3003": "Radiology",
    "X4004": "Medication Charges",
    "X5005": "Surgical Fees",
}

# Extract plain text and OCR fallback

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except:
            pass

        if not text.strip():
            pdf.seek(0)
            images = convert_from_bytes(pdf.read())
            for image in images:
                ocr_results = ocr.ocr(np.array(image), cls=True)
                for line in ocr_results:
                    for word_info in line:
                        text += word_info[1][0] + " "
                    text += "\n"
    return text

# Table extraction from digital PDFs
def extract_tables_from_pdf(pdf_docs):
    tables = []
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table:
                        tables.append(table)
    return tables

# Normalize and map benefit codes
def normalize_tables(tables):
    df_list = []
    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def match_benefits(df):
    if "Code" in df.columns:
        df["Benefit Description"] = df["Code"].map(benefit_codes)
    return df

# Text chunking
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Create vector store
def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text found in the PDF.")
        return
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Groq query

def query_groq_llm(context, question):
    prompt = f"""
You are a bilingual (English + Cantonese) medical document specialist. Whenever you summarize or answer, follow these rules:

1. Speak as an experienced consultant—never mention “AI” or “assistant.”  
2. Deliver your answer under the header **“Summary”**
   - If the user’s question includes “table format”, “tabular”, “in a table”, or similar, return your summary as an **table format**.  
   - Otherwise, provide a detailed  **clean bullet or numbered list** under the header **Summary**, kind of paragraph with each point neatly aligned and labeled.  
3. Produce a concise **bilingual summary**, placing the Cantonese translation immediately in parentheses after each English point.  
4. Expand benefit codes per the list below; unknown codes get “Unknown Code (未知代碼).”  
5. **Error handling**:  
   - If there’s no relevant information in the context, say:  
     "I’m sorry, I couldn’t find any relevant information in the files you provided—I’m limited to the content you’ve uploaded. Please check your uploads or try rephrasing your question. (抱歉，我在您提供的文件中找不到相關資訊——我只能基於您上傳的內容進行回答。請檢查上傳的文件或嘗試重新表述您的問題。)"  
   - If the question is unclear, ask for clarification in both languages.  
   - If the Cantonese translation fails, apologize briefly and continue.

Benefit‑code mappings:
- ACF01: Standard Inpatient Food  
- ACF02: Special Inpatient Food / Medical Nutrition  
- ACF03: Other Special-Purpose Nutrition  
- ACF04: Dietary Supplement  
- ACC13: Mortuary Services  
- MEAM1: Ambulance Services  
- ACS01: Other Hospital Charges  
- ACS02: (Set admit / Gift set)  
- MED00: Other  

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== Summary ===
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
        top_p=1
    )
    return response.choices[0].message.content.strip()
# Process question
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    answer = query_groq_llm(context, user_question)
    st.write("**Data Insights:**", answer)

# Clear input
def clear_question():
    st.session_state.question_input = ""

# Main app
def main():
    st.set_page_config(page_title="DocuGen AI", layout="wide")
    st.markdown("""
        <style>
        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
            background-color: #121212;
            color: #f1f1f1;
        }
        h1, h2, h3 {
            color: #f1f1f1;
        }
        .stTextInput>div>div>input {
            background-color: #1f1f1f;
            color: white;
            border: 1px solid #333;
        }
        .stButton>button {
            background-color: #333;
            color: white;
            border-radius: 6px;
        }
        .stFileUploader {
            border: 1px dashed #666;
            padding: 1rem;
            border-radius: 10px;
            background-color: #1e1e1e;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔎 DocuGen AI")

    col1, col2 = st.columns([9, 1], gap="small")

    with col1:
        user_question = st.text_input(
            "",
            value=st.session_state.get("question_input", ""),
            placeholder="Ask a question from the PDF files",
            key="question_input",
            label_visibility="collapsed",
        )

    with col2:
        st.button("×", key="clear_button", help="Clear input", on_click=clear_question)

    if user_question and user_question.strip():
        user_input(user_question)

    with st.sidebar:
        st.markdown("## 📂 Menu")
        st.markdown("Upload your PDF files and click **Submit & Process**")
        pdf_docs = st.file_uploader(
            "Drag and drop files here",
            accept_multiple_files=True,
            type=["pdf"]
        )
        st.caption("Limit 200MB per file")

        if st.button("📥 Submit & Process"):
            with st.spinner("🔍 Processing your PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                tables = extract_tables_from_pdf(pdf_docs)
                if tables:
                    df = normalize_tables(tables)
                    df = match_benefits(df)
                    st.subheader("📋 Extracted Medical Bill Summary")
                    st.dataframe(df)
                st.success("✅ Processing Complete!")

if __name__ == "__main__":
    main()
