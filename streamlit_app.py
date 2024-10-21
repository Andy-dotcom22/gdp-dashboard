import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import locale

locale.setlocale(locale.LC_ALL, 'en_US')

# File uploaders
uploaded_file_defectos = st.file_uploader("Sube el archivo de Defectos de Ancho", type=["xlsx"])
uploaded_file_demoras = st.file_uploader("Sube el archivo de Demoras", type=["xlsx"])

if uploaded_file_defectos and uploaded_file_demoras:
    try:
        defectos_ancho = pd.read_excel(uploaded_file_defectos, engine='openpyxl')
        demoras_caliente = pd.read_excel(uploaded_file_demoras, engine='openpyxl')

        # Limpieza y preprocesamiento (antes del merge, para mayor eficiencia. Mover al final si se necesita algún dato como string al momento de unir dataframes)
        ID_COILD_1 = defectos_ancho["ID COILD"]
        
        # Configura la localización para .astype(errors='ignore'). Si el formato no coincide con los datos numéricos entonces se producirán errores.
        locale.setlocale(locale.LC_ALL, 'es_ES')  # O 'en_US' u otro, según el formato del archivo

        defectos_ancho['Peso'] = defectos_ancho['Peso'].apply(lambda x: locale.atof(x) if isinstance(x,str) else x).astype(float, errors='ignore')
        defectos_ancho['Fecha'] = pd.to_datetime(defectos_ancho[['YEAR', 'MONTH', 'DAY ']].rename(columns={'DAY ': 'DAY'}), errors='coerce')
        defectos_ancho = defectos_ancho.drop(columns=['MONTH', 'DAY ', 'YEAR', 'Fecha', 'CAIDAS/REPROCESO'])

        demoras_caliente['Duracion'] = demoras_caliente['Duracion'].astype(float, errors='ignore')
        demoras_caliente['Fecha'] = pd.to_datetime(demoras_caliente['Fecha'], format='%d/%m/%Y', errors='coerce')
        demoras_caliente['Mes'] = demoras_caliente['Fecha'].dt.month
        demoras_caliente = demoras_caliente[~demoras_caliente["Fecha"].isna()]
        demoras_caliente = demoras_caliente.drop(columns=["Fecha Inicio", "Fecha Fin"])  # Quitar columnas problemáticas.
       

        # Fusionar tablas (usando merge y un dataframe auxiliar. Mover ambos label encodings *al final* en caso de necesitar un tipo de dato object en el dataframe resultado de esta fusión para hacer merge.)


        defectos_ancho_con_demoras_merged = pd.merge(ID_COILD_1.to_frame(), demoras_caliente, on='ID COILD', how='left')

        defectos_ancho_con_demoras = pd.merge(defectos_ancho, defectos_ancho_con_demoras_merged, left_index=True, right_index=True).drop(columns=["ID COILD_x"])  #Soltar la columna ID_COILD_X

        defectos_ancho_con_demoras = defectos_ancho_con_demoras.rename(columns={col: col+'_right' if col.endswith("_y") else col for col in defectos_ancho_con_demoras.columns})




# Label encoding para variables categóricas. Lo ideal es manejar las object en un inicio si la variable no va a estar involucrada en un merge.  Al hacer merge/join primero debes asegurarte de corregir todos los objetos con los que Pandas no puede trabajar con las opciones que te he dado arriba con locale
# ¡Label encoding debe ir después de unir las tablas! Ya que en este momento se cambian strings por enteros. El label encoding del ejemplo une tablas antes para asegurar eficiencia y no repetir procesos


        label_encoder = LabelEncoder()



        for column in defectos_ancho_con_demoras.columns:

            if defectos_ancho_con_demoras[column].dtype == 'object':



                 defectos_ancho_con_demoras[column] = label_encoder.fit_transform(defectos_ancho_con_demoras[column].astype(str))





       

        defectos_ancho_con_demoras = defectos_ancho_con_demoras.replace([np.inf, -np.inf], np.nan).fillna(0) #Procesando los valores infinitos o np.nan y los convierte en datos con los que Sklearn puede trabajar


#Recuerda que para los selectbox el usuario puede elegir cualquier opción (puede no haber Peso disponible): 
       
       

        # Selección de objetivo y características (Ahora no darán errores si no hay peso en los datos. Modificado para soportar multiselect en variables predictivas/features).

        target_options = [c for c in defectos_ancho_con_demoras.columns if defectos_ancho_con_demoras[c].dtype!=object and c.lower() not in ['index', 'id']]
        objetivo = st.sidebar.selectbox("Variable objetivo", target_options, index=0)


        if objetivo is not None:
          feature_options = [col for col in defectos_ancho_con_demoras.columns if col != objetivo and defectos_ancho_con_demoras[col].dtype!=object]  #Seleccionando características. La mejor forma es con multiselect si tu usuario las puede elegir a voluntad
          caracteristicas = st.sidebar.multiselect("Características", feature_options, default=feature_options)  #Seleccionando características por defecto





        if objetivo and caracteristicas:
            try:


                X = defectos_ancho_con_demoras[caracteristicas].values

                y = defectos_ancho_con_demoras[objetivo].values



        # ... (Resto del código como lo tenías: dividir datos, modelo, predicciones, métricas)



            except Exception as e:

                st.exception(e) #despliega errores



    except Exception as e: #capturando excepciones generales al principio
         st.exception(e)


else:
    st.warning("Sube ambos archivos para comenzar")
