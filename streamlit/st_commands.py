    
import streamlit as st

def main():
    
    st.header('This is a header')
    st.subheader('This is a subheader')
    st.text("It's a text")
    
    st.image('image.png')
    st.audio('record.wav')
    st.video('video.mov')

if __name__ == '__main__':
    main()

#streamlit run doc.py