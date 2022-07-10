# Core Package
from Model import mobile_phone
from Controller import ml_model
import streamlit as st
import pickle

# Exploratory data Analisis Packages
import pandas as pd
import numpy as np

# Visualization Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
        page_title="Data Science Exploration",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )


def main():
    st.title("Data Science Analisis App")
    st.text("Ths is Aplication My portofolio project for Data Scientist")
    activities = ["Exploratory DA", "Visualization",
                  "Machine Learning Performance", "App Demonstration"]
    choice = st.sidebar.selectbox("Select Activities", activities)

    if choice == 'Exploratory DA':
        st.subheader("Exploratory data Analysis")
        df = pd.read_csv("train.csv")
        st.dataframe(df.head())

        # Semua baris
        if st.checkbox("Show All Rows"):
            st.dataframe(df)
         # Jumlah kolom dan baris
        if st.checkbox("Show Columns"):
            all_colom = df.columns.to_list()
            st.dataframe(all_colom)
        # Corelasi Antar kolom
        if st.checkbox("Show Corelation Feature"):
            st.dataframe(df.corr())
        # Pilih kolom korelasi
        if st.checkbox("Select columns to Show corelation"):
            all_colom = df.columns.to_list()
            selected_columns = st.multiselect("Select Columns", all_colom)
            df_sel_cor = df[selected_columns]
            st.dataframe(df_sel_cor.corr())
        # Kolom target
        if st.checkbox("Show Target Columns"):
            st.dataframe(df["price_range"].value_counts())
        # Korelasi tertinggi
        if st.checkbox("Show Hight Corelation"):
            df_corr = df[['battery_power', 'ram',
                          'px_width', 'px_height', 'price_range']]
            st.dataframe(df_corr.corr())
        # Deskriptif Statistik
        if st.checkbox("Show Descriptive Statistics"):
            st.dataframe(df.describe())
            st.write(
                "In the columns PX_HEIGHT there is a strange value and will be deleted")
            st.dataframe(df.loc[df['px_height'] == 0])
        # Total baris dan Kolom
        if st.checkbox("Show Feature Count"):
            st.dataframe(df.shape)
        # data bersih
        if st.checkbox("Show Clean Dataset Training"):
            df_clean = pd.read_csv("data_bersih.csv")
            df_clean.drop("Unnamed: 0", axis=1, inplace=True)
            st.dataframe(df_clean.head())
            st.write(df_clean.shape)

    # Visualisasi
    elif choice == 'Visualization':
        st.subheader("Data Visualization")

        # data yang digunakan
        df = pd.read_csv("train.csv")

        # Pie chart
        if st.checkbox("Pie Chart"):
            all_colom = df.columns.to_list()
            columns_to_plot = st.selectbox("Select 1 Columns", all_colom)
            pie_plot = df[columns_to_plot].value_counts().plot.pie(
                autopct='%1.1f%%')
            st.write(pie_plot)
            st.pyplot()

        # Pilihan untuk plot
        all_columns_names = df.columns.to_list()
        type_of_plot = st.selectbox(
            "Select Type Plot", ["Histogram", "Line", "Box", "Violin"])
        selected_columns_names = st.multiselect(
            "Select Columns To Plot", all_columns_names)
        if st.button("Click To Plot"):
            st.success("Generating Customization Plot Of {} for {}".format(
                type_of_plot, selected_columns_names))

             # Buat sendiri lah ya kali kopas kopas
            if type_of_plot == "Histogram":
                cust_data = df[selected_columns_names]
                fig, ax = plt.subplots()
                ax.hist(cust_data, bins=30, color='green')
                st.pyplot(fig)

            elif type_of_plot == "Line":
                cust_data = df[selected_columns_names]
                st.line_chart(cust_data)

            elif type_of_plot == "Box":
                cust_data = df[selected_columns_names]
                fig, ax = plt.subplots()
                ax.boxplot(cust_data)
                st.pyplot(fig)
            elif type_of_plot == "Violin":
                cust_data = df[selected_columns_names]
                fig, ax = plt.subplots()
                ax.violinplot(cust_data)
                st.pyplot(fig)

            elif type_of_plot:
                cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()

    # Machine learning performa
    if choice == 'Machine Learning Performance':
        st.subheader("Evaluation Machine Learning Model")
        ml_model.perform()
        
    # App Demonstrasi
    if choice == 'App Demonstration':
        mobile_phone.Model()


if __name__ == '__main__':
    main()
