import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3
import streamlit.components.v1 as components
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st

df = pd.read_csv('heart.csv')
st.set_page_config(layout='wide')
st.sidebar.header("Graphs")


st.title("Raw Heart.csv data")
st.write(df)

# first column line
st.title('Graphs')
st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(1)
        {
            border:0px solid red;
        } 

        div[data-testid="column"]:nth-of-type(2)
        {
            border:0px solid blue;
            text-align: left;
        } 
    </style>
    """, unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.title('Age vs Cholesterol')
    age_bar = alt.Chart(df).mark_bar().encode(
        x=alt.X('Age'),
        y=alt.Y('Cholesterol'),
    ).interactive()
    st.altair_chart(age_bar)

with col2:
    st.title('Age vs RestingBP')
    age_scatter = alt.Chart(df).mark_square().encode(
        x=alt.X('Age', scale=alt.Scale(domain=(20, 80))),
        y=alt.Y('RestingBP'),
    ).interactive()
    st.altair_chart(age_scatter)
with col3:
    st.title('Age vs MaxHR')

    step=alt.Chart(df).mark_line().encode(
        x='Age',
        y='MaxHR'
    ).interactive()
    st.altair_chart(step)
# second column graph
st.markdown(
    """
    <style>
        div[data-testid="column"]:nth-of-type(1)
        {
            border:0px solid red;
        } 

        div[data-testid="column"]:nth-of-type(2)
        {
            border:0px solid blue;
            text-align: left;
        } 
    </style>
    """, unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    st.title('Total ChestPaintType')
    df_g = df.groupby(['ChestPainType'])['ChestPainType'].count()
    df_group = df_g.to_frame()
    df_group['Types'] = df_group.index
    base = alt.Chart(df_group).encode(
        theta=alt.Theta("ChestPainType:Q", stack=True),
        radius=alt.Radius("ChestPainType", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color="Types:N",
    )

    c1 = base.mark_arc(innerRadius=20, stroke="#fff")

    c2 = base.mark_text(radiusOffset=10).encode(text="ChestPainType:Q")
    layer = alt.layer(c1, c2)
    st.altair_chart(layer)

with col2:
    st.title('Total ST_Slope')
    percentage = alt.Chart(df).transform_aggregate(
        total='count(ST_Slope)',
        groupby=['ST_Slope'],
    ).mark_bar().encode(
        alt.Y('ST_Slope:N'),
        alt.X('total:Q'),
    ).properties(
        width=450,
        height=350
    ).interactive()
    st.altair_chart(percentage)
with col3:
    st.title('Total Sex Distribution')
    a=alt.Chart(df).transform_aggregate(
        total='count(Sex)',
        groupby=['Sex'],
    ).mark_arc().encode(
        theta=alt.Theta(field="total", type="quantitative"),
        color=alt.Color(field="Sex", type="nominal"),
    ).interactive()
    st.altair_chart(a)

