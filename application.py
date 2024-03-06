import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare
import csv
from transformers import pipeline


# def extract_pdf_data(file_path):
#     data = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 data += text
#     return data


# def extract_text_data(file_path):
#     with open(file_path, 'r') as file:
#         data = file.read()
#     return data


# # Command-line argument processing
# if len(sys.argv) > 1:

#     if len(sys.argv) == 3:
#         resume_path = sys.argv[1]
#         jd_path = sys.argv[2]

#         resume_data = extract_pdf_data(resume_path)
#         jd_data = extract_text_data(jd_path)

#         result = compare([resume_data], jd_data, flag='HuggingFace-BERT')

#     sys.exit()


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Sidebar
tab1,tab2 = st.tabs(["**Data Awal**", "**Hasil**"])
# Tab Home
with tab1:
    st.title("Application Sumarry Text")
    # uploaded_files = st.file_uploader(
    #     '**Choose your resume.pdf file:** ', type="pdf", accept_multiple_files=True)
    summary = None
    text = st.text_area("**Enter text :**")
    comp_pressed = st.button("Compare!")
    if comp_pressed :
        summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
        # print(summary[0]['summary_text'])
st.write("Hasil")
if summary:
    st.write(summary[0]['summary_text'])

