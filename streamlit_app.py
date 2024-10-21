import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Título
st.title("Análisis de Defectos de Ancho en MC4")

# File uploaders
uploaded_file_defectos = st.file_uploader("Sube el archivo de Defectos de Ancho", type=["xlsx"])
uploaded_file_demoras = st.file_uploader("Sube el archivo de Demoras", type=["xlsx"])

if uploaded_file_defectos and uploaded_file_demoras:
    try:
        defectos_ancho = pd.read_excel(uploaded_file_defectos, engine='openpyxl')
        demoras_caliente = pd.read_excel(uploaded_file_demoras, engine='openpyxl')

        # Limpieza de defectos_ancho
        ID_COILD_1 = defectos_ancho["ID COILD"]
        defectos_ancho['Peso'] = defectos_ancho['Peso'].astype(float, errors='ignore')
        defectos_ancho['Fecha'] = pd.to_datetime(defectos_ancho[['YEAR', 'MONTH', 'DAY ']].rename(columns={'DAY ': 'DAY'}), errors='coerce')
        defectos_ancho = defectos_ancho.drop(columns=['MONTH', 'DAY ', 'YEAR', 'Fecha'])  # Corregido: Eliminamos la columna 'Fecha' una vez

        # Limpieza de demoras_caliente
        demoras_caliente['Duracion'] = demoras_caliente['Duracion'].astype(float, errors='ignore')
        demoras_caliente['Fecha'] = pd.to_datetime(demoras_caliente['Fecha'], format='%d/%m/%Y', errors='coerce')
        demoras_caliente['Fecha Inicio'] = pd.to_datetime(demoras_caliente['Fecha Inicio'], errors='coerce')
        demoras_caliente['Fecha Fin'] = pd.to_datetime(demoras_caliente['Fecha Fin'], errors='coerce')
        demoras_caliente['Mes'] = demoras_caliente['Fecha'].dt.month

        # Fusionar tablas
        defectos_ancho_con_demoras_merged = pd.merge(ID_COILD_1, demoras_caliente, on='ID COILD', how='left')

        defectos_ancho_con_demoras = pd.merge(defectos_ancho, defectos_ancho_con_demoras_merged, left_index=True, right_index=True)

        # Preprocesamiento: Label Encoding
        label_encoder = LabelEncoder()
        for column in defectos_ancho_con_demoras.columns:
           if defectos_ancho_con_demoras[column].dtype == 'object':

                defectos_ancho_con_demoras[column] = label_encoder.fit_transform(defectos_ancho_con_demoras[column].astype(str))  # apply label encoding


        defectos_ancho_con_demoras.fillna(0, inplace=True)  # Reemplaza valores infinitos e infinitamente pequeños y los convierte en np.nan. Para finalizar reemplazamos todos los nan por 0

        # Selección de objetivo (En la barra lateral)
        st.sidebar.header("Selecciona el objetivo")
        target_options = defectos_ancho_con_demoras.columns.tolist()
        objetivo = st.sidebar.selectbox("Variable Objetivo", target_options, index=target_options.index('Peso'))

        # Selección de características (En la barra lateral, todas menos el objetivo)
        st.sidebar.header("Selecciona las características")
        feature_options = [col for col in target_options if col != objetivo]
        caracteristicas = st.sidebar.multiselect("Características", feature_options, default=feature_options)


        if objetivo and caracteristicas:  # Verificar que se seleccionó un objetivo y al menos una característica
            try:

                X = defectos_ancho_con_demoras[caracteristicas]

                y = defectos_ancho_con_demoras[objetivo]



                # Dividir datos
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


                # Modelo
                modelo = LinearRegression()  # Considera la posibilidad de cambiar a otros modelos si lo crees oportuno.

                modelo.fit(X_train, y_train)


                # Predicciones
                predicciones = modelo.predict(X_test)


                # Métricas

                mse = mean_squared_error(y_test, predicciones)
                r2 = r2_score(y_test, predicciones)


                st.header("Resultados del Modelo")
                st.write(f"Error Cuadrático Medio (MSE): {mse}")
                st.write(f"R-cuadrado (R2): {r2}")


                # Mostrar predicciones

                mostrar_predicciones = st.checkbox("Mostrar Predicciones")

                if mostrar_predicciones:
                    predicciones_df = pd.DataFrame({"Real": y_test, "Predicción": predicciones})
                    st.write(predicciones_df)


            except Exception as e: #Captura la gran mayoria de errores de tipo de dato o ajuste en general
                st.exception(e) #Mostrar errores en la vista principal



    except Exception as e: #Capturar excepciones por el input de pandas, por datos no reconocidos
        st.exception(e)  # Mostrar excepciones de lectura del archivo

else:
    st.warning("Sube ambos archivos Excel para comenzar.")
