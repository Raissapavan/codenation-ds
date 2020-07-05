import pandas as pd
import streamlit as st
import altair as alt

def criar_histograma(coluna, df):
    chart = alt.chart(df, width=600).mark_bar.encode(
        alt.X(coluna, bin=True),
        y='count()', tooltip=[coluna, 'count()']
    ).interactive()
    return chart

def main():
    

if __name__ == '__main__':
    main()