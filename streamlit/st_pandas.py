import pandas as pd
import streamlit as st

def main():
    st.title('AceleraDev Data Science')

    file = st.file_uploader('Upload your file', type = 'csv')
    if file is not None:
        slider = st.slider('Valores', 1,100)
        df = pd.read_csv(file)
        st.dataframe(df.head(slider))
        st.markdown('Table format')
        st.table(df.head(slider))
        st.write(df.columns)
        st.table(df.groupby('Age')['Gender'].count())

if __name__ == '__main__':
    main()