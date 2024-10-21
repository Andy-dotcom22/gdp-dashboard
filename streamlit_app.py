pip install -r requirements.txt
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Título
st.title("Análisis de Defectos de Ancho en MC4")

# Carga de datos
defectos_ancho = pd.read_excel("C:/Users/Andrea Santos/Documents/7mo Semestre/Reto/2nd Period/00-Defectos Ancho - Caidas & Reprocesos - MC4.xlsx")
demoras_caliente = pd.read_excel("C:/Users/Andrea Santos/Documents/7mo Semestre/Reto/2nd Period/SGL_Demoras_Caliente 4 PES.xlsx")


# Limpieza de defectos_ancho
defectos_ancho['Peso'] = defectos_ancho['Peso'].str.replace(',', '.', regex=False).astype(float, errors='ignore')
defectos_ancho['Fecha'] = pd.to_datetime(defectos_ancho[['YEAR', 'MONTH', 'DAY ']].rename(columns={'DAY ': 'DAY'}), errors='coerce')
defectos_ancho = defectos_ancho.drop(columns=['MONTH', 'DAY ', 'YEAR', 'Fecha', 'CAIDAS/REPROCESO'])

# Limpieza de demoras_caliente
demoras_caliente['Duracion'] = demoras_caliente['Duracion'].astype(float, errors='ignore') # Ya no es necesario convertir la coma, la detecta como flotante de forma automática
demoras_caliente['Fecha'] = pd.to_datetime(demoras_caliente['Fecha'], format='%d/%m/%Y', errors='coerce') # Cambié Y%m%d por %d/%m/%Y debido al formato de la fecha en el archivo, para mayor flexibilidad también añadí error='coerce'
demoras_caliente['Fecha Inicio'] = pd.to_datetime(demoras_caliente['Fecha Inicio'], errors='coerce')
demoras_caliente['Fecha Fin'] = pd.to_datetime(demoras_caliente['Fecha Fin'], errors='coerce')
demoras_caliente['Mes'] = demoras_caliente['Fecha'].dt.month

# Fusionar tablas
defectos_ancho_con_demoras = pd.merge(defectos_ancho, demoras_caliente, on='ID COILD', how='left')

# Preprocesamiento: Label Encoding para variables categóricas
label_encoder = LabelEncoder()
for column in defectos_ancho_con_demoras.columns:
    if defectos_ancho_con_demoras[column].dtype == 'object':
        defectos_ancho_con_demoras[column] = label_encoder.fit_transform(defectos_ancho_con_demoras[column])

# Manejar valores infinitos o NaN después de la codificación. Reemplazarlos con 0 es una opción sencilla, pero considera estrategias más sofisticadas según tu caso.
defectos_ancho_con_demoras = defectos_ancho_con_demoras.replace([np.inf, -np.inf], np.nan).fillna(0)

# Selección de características y objetivo
st.sidebar.header("Selecciona características y objetivo")
target_options = defectos_ancho_con_demoras.columns.tolist()
objetivo = st.sidebar.selectbox("Variable Objetivo", target_options, index=target_options.index("Peso"))  # Peso por defecto

if objetivo:
    try:

        X = defectos_ancho_con_demoras.drop(columns=[objetivo]).astype(float)
        y = defectos_ancho_con_demoras[objetivo].astype(float)

        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo (Regresión Lineal)
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Predicciones
        predicciones = modelo.predict(X_test)

        # Métricas
        mse = mean_squared_error(y_test, predicciones)
        r2 = r2_score(y_test, predicciones)

        st.header("Resultados del Modelo")
        st.write(f"Error Cuadrático Medio (MSE): {mse}")
        st.write(f"R-cuadrado (R2): {r2}")


        # Mostrar predicciones (opcional)
        mostrar_predicciones = st.checkbox("Mostrar Predicciones")
        if mostrar_predicciones:
           predicciones_df = pd.DataFrame({"Real": y_test, "Predicción": predicciones})
           st.write(predicciones_df)

    except ValueError as e: # Es posible que float no reconozca ciertos datos.
        st.error(f"Error de valor:\n {e}.\n Por favor, verifica la integridad del archivo para datos faltantes/inconsistencias.")
