import pandas as pd
import altair as alt
import numpy as np
import streamlit as st
from scipy.stats import norm, stats

df = pd.read_csv('nobel_prize_by_winner.csv')
st.set_page_config(layout='wide')
st.markdown("""
# Statistical Analysis
""")
tab1, tab2 = st.tabs(["Raw Data", "Prepared data"])

with tab1:
    st.header("Raw Data")
    st.write(df)

with tab2:
    df.drop(
        ['index', 'overallMotivation', 'bornCity', 'diedCity', 'diedCountry', 'diedCountryCode',
         'share', 'motivation', 'bornCountryCode', 'name', 'city'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df['firstname'].dropna(inplace=True)
    df["Full Name"] = df['firstname'].astype(str) + " " + df["surname"].astype(str)
    df.drop(['firstname', 'surname'], axis=1, inplace=True)
    df.dropna(inplace=True)
    df['Alive'] = np.where(df['died'] == '0000-00-00', 'Alive', 'Dead')
    index = df[(df['born'] == '0000/00/00')].index
    df.drop(index, inplace=True)
    df['born'] = df['born'].astype('string')
    df['year'] = df['year'].astype('float')
    df['born'] = df['born'].str[-4:]
    df['born'] = df['born'].astype('float')
    df['Age'] = (df['year'] - df['born'])
    df['Age'] = df['Age'].astype('int')
    df.drop('id', axis=1, inplace=True)
    df.reset_index()
    df = df.reindex(['Full Name', 'gender', 'Age', 'bornCountry', 'category', 'country',
                     'year', 'born', 'died', 'Alive'], axis=1)
    st.write(df)
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

col1, col2, col3= st.columns(3)

with col1:
    st.header('Total Winners by Age')
    age_total = alt.Chart(df).transform_aggregate(
        total='count(Age)',
        groupby=['Age'],
    ).mark_bar().encode(
        alt.X('Age:N'),
        alt.Y('total:Q'),
        color=alt.value('#39E0F4'),
    ).properties(
        width=450,
        height=350
    ).interactive()
    st.altair_chart(age_total)
    st.markdown("""
    Graph depicts relationship between age of Nobel Prize Winners and Total Prize per age.
        """)
with col2:
    st.header('Total Winners by Age Group')
    age_groups = alt.Chart(df).mark_bar().encode(
        alt.X("Age", bin=alt.Bin(extent=[10, 100], step=10)),
        y='count()',
        color=alt.value('#39E0F4'),
    ).properties(
        width=450,
        height=350
    ).interactive()

    rule = alt.Chart(df).mark_rule(color='red').encode(
        x='mean(Age):Q',
        size=alt.value(3)
    )
    line_graph = alt.Chart(df).mark_line().encode(
        alt.X("Age", bin=alt.Bin(extent=[0, 100], step=10)),
        y='count()',
        color=alt.value('#2300FF'),
        size=alt.value(3)

    )
    layer = alt.layer(age_groups, rule, line_graph)
    st.altair_chart(layer)
    st.markdown("""
    Graph shows the same data for age group rather than each age. It help us to see normal distribution line better. 
    Red line refers to mean of age. We can see from the 'bell curved' line, 
    data is distributed  similarly from mean point in both direction.
    """)
with col3:
    st.header('Age vs Z Values')
    df['zscores'] = stats.zscore(df['Age'])
    df['Norm'] = norm.cdf(df['Age'])
    zscore = alt.Chart(df).mark_line().encode(
        x=alt.X('Age', scale=alt.Scale(domain=(20, 100))),
        y=alt.Y('zscores', scale=alt.Scale(domain=(-3, 3)))
    )
    st.altair_chart(zscore)
    st.markdown("""
        Graph shows the relationship between Age and Z values of those ages. 
        We can see mean of ages approximately correspond to 0 value in Z values.
        """)
st_dev = round(np.std(df['Age']),2)
mean_df_age = round(np.mean(df["Age"]),2)
coef_var = round(st_dev/mean_df_age,2)

html_str = f"""
<style>
p.a {{
  font: bold 25px Courier;
}}
</style>
<p class="a">σ = {st_dev}: Standard Deviation<br>
µ = {mean_df_age} : Mean<br>
CV= {coef_var} : Coefficient of Variation</p>
"""

st.markdown(html_str, unsafe_allow_html=True)
st.markdown("""
Since coefficient of variation is less than 1, it indicates that the standard deviation is smaller than the mean
""")




