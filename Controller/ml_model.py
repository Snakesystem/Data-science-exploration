import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns 
sns.set()


def perform():
    df = pd.read_csv('train.csv')

    df_train = df[['battery_power', 'ram', 'int_memory', 'px_width', 'px_height', 'price_range']]

    df_new = df_train.copy()
    df_new.drop([1481,1933], inplace=True)
    df_new.reset_index(drop=True, inplace=True)

    X = df_new.loc[:, df_new.columns != 'price_range']
    y = df_new["price_range"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    
    rf = pickle.load(open('mobile_rf_performance.pkl', 'rb'))
    
    y_rf = rf.predict(X_test)
   
    st.subheader('Classification Report From Model')
    st.text(classification_report(y_test, y_rf))
    
    # Visualisasi Confusion Matrix
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, cmap='YlGnBu')
    st.pyplot(fig)


            

if __name__ == '__main__':
    
    perform()