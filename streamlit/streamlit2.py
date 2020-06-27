import pandas as pd
import streamlit as st
import base64

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    return href

def main():
    st.title('AceleraDev Data Science')
    st.subheader('Semana 2 - Pré Processamento de Dados')

    st.markdown('File Uploader')
    file = st.file_uploader('Choose your file', type = 'csv')
    if file is not None:
        st.subheader('Analisando Dados')
        df = pd.read_csv('file')
        st.markdown('Número de linhas:')
        st.markdown(df.shape[0])
