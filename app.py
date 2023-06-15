import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import pickle5 as pickle 

st.set_page_config(
    page_title="Prediksi Saham PT.Indosat Ooredoo Hutchison Tbk",
    page_icon='https://icon-library.com/images/data-science-icon/data-science-icon-25.jpg',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""<h1>Aplikasi Prediksi Saham PT.Indosat Ooredoo Hutchison Tbk</h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h2 style = "text-align: center;"><img src="https://icon-library.com/images/data-science-icon/data-science-icon-25.jpg" width="130" height="130"><br></h2>""",unsafe_allow_html=True), 
        ["Home", "Data", "Prepocessing", "Modeling", "Implementation"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#005980"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#005980"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Indosat_Ooredoo.svg/1280px-Indosat_Ooredoo.svg.png" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Data":
        st.subheader("""Deskripsi Aplikasi""")
        st.write("""
         Aplikasi Indosat Prediction merupakan aplikasi yang digunakan untuk memprediksi saham dari PT.Indosat Ooredoo pada hari berikutnya. 
        """)

        st.subheader("""Deskripsi Data""")
        st.write("""
        Data yang digunakan dalam aplikasi ini yaitu data saham PT.Indosat Ooredoo Hutchison Tbk periode 15 Juni 2022 sampai 15 Juni 2023. Data yang ditampilkan adalah data saham yang diperoleh per harinya. 
        """)

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber data didapatkan dari Yahoo Finance. Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://finance.yahoo.com/quote/ISAT.JK/history?p=ISAT.JK">Klik disini</a>""", unsafe_allow_html=True)
        
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset saham PT.Indosat Ooredoo Hutchison Tbk ini adalah NUMERICAL.
        """)

        st.subheader("""Dataset Saham PT.Indosat Ooredoo Hutchison Tbk""")
        df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/ISAT.JK.csv')
        st.dataframe(df, width=600)

    elif selected == "Prepocessing":
        st.subheader("""Univariate Transform""")
        uni = pd.read_csv('dfvolume_unvariate.csv')
        uni = uni.iloc[:, 1:9]
        st.dataframe(uni)
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaledX = scaler.fit_transform(uni)
        features_namesX = uni.columns.copy()
        #features_names.remove('label')
        scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

        st.subheader('Hasil Normalisasi Data')
        st.dataframe(scaled_featuresX.iloc[:,0:7], width=600)

    elif selected == "Modelling":

        uni = pd.read_csv('dfvolume_unvariate.csv')
        uni = uni.iloc[:, 1:9]

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaledX = scaler.fit_transform(uni)
        features_namesX = uni.columns.copy()
        #features_names.remove('label')
        scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

        #Split Data 
        training, test = train_test_split(scaled_featuresX.iloc[:,0:7],test_size=0.1, random_state=0,shuffle=False)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(scaled_featuresX.iloc[:,-1], test_size=0.1, random_state=0,shuffle=False)#Nilai Y training dan Nilai Y testing


        st.write("#### Percobaan Model")
        st.markdown("""
        Dimana :
        - Jumlah Fitur Transform Univariet = [1,2,3,4,5] 
        - K = [3,5,7,9,11]
        - Test Size = [0.2,0.3,0.4]
        """)
        df_percobaan = pd.read_csv('hasil_percobaan1.csv')
        st.write('##### Hasil :')
        data = pd.DataFrame(df_percobaan.iloc[:,1:6])
        st.write(data)
        st.write('##### Grafik Pencarian Nilai Error Terkecil :')
        st.line_chart(data=data[['Nilai Error MSE','Nilai Error MAPE']], width=0, height=0, use_container_width=True)
        st.write('##### Best Model :')
        st.info("Jumlah Fitur = 7, K = 3, Test_Size = 0.1, Nilai Erorr MSE= 0.0133, Nilai Error MAPE = 0,085")
        st.write('##### Model KNN :')

        # load saved model
        with open('model_knn_pkl' , 'rb') as f:
            model = pickle.load(f)
        regresor = model.fit(training, training_label)
        st.info(regresor)

            

    elif selected == "Implementation":
        with st.form("Implementation"):
            uni = pd.read_csv('dfvolume_unvariate.csv')
            uni = uni.iloc[:, 1:9]

            scaler = MinMaxScaler()
            #scaler.fit(features)
            #scaler.transform(features)
            scaledX = scaler.fit_transform(uni)
            features_namesX = uni.columns.copy()
            #features_names.remove('label')
            scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

            #Split Data 
            training, test = train_test_split(scaled_featuresX.iloc[:,0:7],test_size=0.1, random_state=0,shuffle=False)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(scaled_featuresX.iloc[:,-1], test_size=0.1, random_state=0,shuffle=False)#Nilai Y training dan Nilai Y testing

            #Modeling
            # load saved model
            with open('model_knn_pkl' , 'rb') as f:
                model = pickle.load(f)
            regresor = model.fit(training, training_label)
            pred_test = regresor.predict(test)
            
            #denomalize data test dan predict
            hasil_denormalized_test = []
            for i in range(len(test)):
                df_min = uni.iloc[:,0:7].min()
                df_max = uni.iloc[:,0:7].max()
                denormalized_data_test_list = (test.iloc[i]*(df_max - df_min) + df_min).map('{:.1f}'.format)[0]
                hasil_denormalized_test.append(denormalized_data_test_list)

            hasil_denormalized_predict = []
            for y in range(len(pred_test)):
                df_min = uni.iloc[:,0:7].min()
                df_max = uni.iloc[:,0:7].max()
                denormalized_data_predict_list = (pred_test[y]*(df_max - df_min) + df_min).map('{:.1f}'.format)[0]
                hasil_denormalized_predict.append(denormalized_data_predict_list)

            denormalized_data_test = pd.DataFrame(hasil_denormalized_test,columns=["Testing Data"])
            denormalized_data_preds = pd.DataFrame(hasil_denormalized_predict,columns=["Predict Data"])

            #Perhitungan nilai error
            MSE = mean_squared_error(test_label,pred_test)
            MAPE = mean_absolute_percentage_error(denormalized_data_test,denormalized_data_preds)

            # st.subheader("Implementasi Prediksi ")
            v1 = st.number_input('Masukkan Jumlah volume saham pada 7 hari sebelum periode yang akan di prediksi')
            v2 = st.number_input('Masukkan Jumlah volume saham pada 6 hari sebelum periode yang akan di prediksi')
            v3 = st.number_input('Masukkan Jumlah volume saham pada 5 hari sebelum periode yang akan di prediksi')
            v4 = st.number_input('Masukkan Jumlah volume saham pada 4 hari sebelum periode yang akan di prediksi')
            v5 = st.number_input('Masukkan Jumlah volume saham pada 3 hari sebelum periode yang akan di prediksi')
            v6 = st.number_input('Masukkan Jumlah volume saham pada 2 hari sebelum periode yang akan di prediksi')
            v7 = st.number_input('Masukkan Jumlah volume saham pada 1 hari sebelum periode yang akan di prediksi')

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                    v6,
                    v7
                ])
                
                df_min = uni.iloc[:,0:7].min()
                df_max = uni.iloc[:,0:7].max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                st.write("#### Normalisasi data Input")
                st.write(input_norm)

                input_pred = regresor.predict(input_norm)

                st.write('#### Hasil Prediksi')
                st.info((input_pred*(df_max - df_min) + df_min).map('{:.1f}'.format)[0])
                st.write('#### Nilai Error')
                st.write("###### MSE :",MSE)
                st.write("###### MAPE :",MAPE)
