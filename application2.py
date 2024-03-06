import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare
import csv
import pandas as pd
import matplotlib.pyplot as plt 

import seaborn as sns
import cv2

def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data


def extract_text_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


# Command-line argument processing
if len(sys.argv) > 1:

    if len(sys.argv) == 3:
        resume_path = sys.argv[1]
        jd_path = sys.argv[2]

        resume_data = extract_pdf_data(resume_path)
        jd_data = extract_text_data(jd_path)

        result = compare([resume_data], jd_data, flag='HuggingFace-BERT')

    sys.exit()

# Sidebar
flag = 'HuggingFace-BERT'
with st.sidebar:
    st.markdown('**Pilih Teknik Embedding yang digunakan**')
    options = st.selectbox('Sialkan Dipilih',
                           ['HuggingFace-BERT', 'Doc2Vec'],
                           label_visibility="collapsed")
    flag = options

# Main content
tab1, tab2 = st.tabs(["**Data Awal**", "**Hasil**"])

# Tab Home
with tab1:
    st.title("Resume and CV Screening System")
    uploaded_files = st.file_uploader(
        '**Masukan File CV dalam bentuk.pdf file:** ', type="pdf", accept_multiple_files=True)
    JD = st.text_area("**Masukan Uraian Pekerjaan:**")
    comp_pressed = st.button("Proses Membandingkan!")
    if comp_pressed and uploaded_files:
        # Streamlit file_uploader gives file-like objects, not paths
        uploaded_file_paths = [extract_pdf_data(
            file) for file in uploaded_files]
        score = compare(uploaded_file_paths, JD, flag)

# Tab Results
with tab2:
    st.header("Hasil")
    my_dict = {}
    if comp_pressed and uploaded_files:
        for i in range(len(score)):
            my_dict[uploaded_files[i].name] = score[i]
        sorted_dict = dict(sorted(my_dict.items()))
        for i in sorted_dict.items():
            with st.expander(str(i[0])):
                st.write("Score is: ", i[1])
        
        with open('mycsvfile.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
            w = csv.DictWriter(f, my_dict.keys())
            w.writeheader()
            w.writerow(my_dict)

        df = pd.DataFrame(my_dict,index=[0])
        df1=df.T.reset_index().T
        df_transposed = df1.T
        df_transposed.columns = ['Name', 'Sim_Index']
        df_transposed['Sim_Index'] = df_transposed['Sim_Index'].astype('float')
        yellow='#FFB11E'
        by_smm=sns.barplot(x ='Name',y ='Sim_Index',data = df_transposed,color=yellow)
        by_smm.set_xticklabels(by_smm.get_xticklabels(),rotation=45, horizontalalignment='right')
        by_smm.bar_label(by_smm.containers[0])
        by_smm.axhline(y = 75,xmax = 100) # To the right
        # Display the plot in Streamlit
        st.pyplot(by_smm.get_figure())

        score_model =df_transposed.to_csv(index=False).encode('utf-8')
        st.download_button(
        label="Download data as CSV",
        data=score_model,
        mime='text/csv')





