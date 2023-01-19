import pandas as pd
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import streamlit as st

df = pd.read_csv('heart.csv')
true_values = df['HeartDisease']
st.set_page_config(layout='wide',
                   page_icon='🧠')
st.markdown("""
# Machine Learning Algorithm Performances
Heart Diseases dataset is used.
This data set dates from 1988 and consists of four databases: Cleveland, Hungary, Switzerland, and Long Beach V. 
It contains 76 attributes, including the predicted attribute, 
but all published experiments refer to using a subset of 14 of them.
The "target" field refers to the presence of heart disease in the patient. 
It is integer valued 0 = no disease and 1 = disease.
""")
tab1, tab2 = st.tabs(["Raw Data", "Processed data"])

with tab1:
    st.header("Raw Data")
    st.write(df)

with tab2:
    df['Sex'].replace('M', 0, inplace=True)
    df['Sex'].replace('F', 1, inplace=True)
    df['ExerciseAngina'].replace('N', 0, inplace=True)
    df['ExerciseAngina'].replace('Y', 1, inplace=True)
    df['RestingECG'].replace('Normal', 0, inplace=True)
    df['RestingECG'].replace('ST', 1, inplace=True)
    df['RestingECG'].replace('LVH', 2, inplace=True)
    df['ST_Slope'].replace('Up', 0, inplace=True)
    df['ST_Slope'].replace('Flat', 1, inplace=True)
    df['ST_Slope'].replace('Down', 2, inplace=True)
    df['ChestPainType'].replace('ATA', 0, inplace=True)
    df['ChestPainType'].replace('NAP', 1, inplace=True)
    df['ChestPainType'].replace('ASY', 2, inplace=True)
    df['ChestPainType'].replace('TA', 3, inplace=True)
    st.header("Data after preprocessed")
    st.write(df)

X = df.iloc[:, 0:11]
y = df.iloc[:, 11]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
s_X = StandardScaler()
X_train = s_X.fit_transform(X_train)
X_test = s_X.transform(X_test)
classifier4 = KNeighborsClassifier(n_neighbors=29, metric='euclidean', p=2)
classifier4.fit(X_train, y_train)
y_pred = classifier4.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
confusion_data = pd.DataFrame(confusion)
y_pred_data = pd.DataFrame(y_pred,columns=['Prediction'])

# NaiveBayes
classifier = GaussianNB()
classifier.fit(X_train, y_train)


y_pred_nb = classifier.predict(X_test)

confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)


st.title('K-Nearest Neighbor Model Result')
# second column line
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
            text-align: center;
        } 
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.header('Confusion matrix')
    st.write(confusion_data)
with col2:
    result = str(round(accuracy_score(y_test, y_pred), 2))
    st.header('Accuracy Result')
    st.metric(label="Percentage", value=result+"%")

# NaiveBayes
st.title('Gaussian Naive Bayes Model Result')
# second column line
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
            text-align: center;
        } 
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.header('Confusion matrix')
    st.write(confusion_matrix_nb)
with col2:
    result =str(round(accuracy_score(y_test, y_pred_nb), 2))
    st.header('Accuracy Result')
    st.metric(label="Percentage", value=result+"%")


classifier1 = LogisticRegression(random_state=0)
classifier1.fit(X_train, y_train)
y_pred_lr = classifier1.predict(X_test)
confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)


# Linear Regression
st.title('Linear Regression Model Result')
# second column line
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
            text-align: center;
        } 
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.header('Confusion matrix')
    st.write(confusion_matrix_lr)
with col2:
    result = str(round(accuracy_score(y_test, y_pred_lr), 2))
    st.header('Accuracy Result')
    st.metric(label="Percentage", value=result+"%")
# Decision Trees

classifier2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier2.fit(X_train, y_train)

y_pred_dt = classifier2.predict(X_test)

confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)
st.title('Decision Tree Model Result')

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

col1, col2 = st.columns(2)

with col1:
    st.header('Confusion matrix')
    st.write(confusion_matrix_dt)
with col2:
    result = str(round(accuracy_score(y_test, y_pred_dt), 2))
    st.header('Accuracy Result')
    st.metric(label="Percentage", value=result+"%")

