import streamlit as st
import pandas as pd
import os
import time
import datetime

# Configuración de la página
st.set_page_config(page_title="Dashboard de Bases de Datos", layout="wide")

# Función para leer datos (adaptada para manejar el formato de tu archivo)
@st.cache_data(ttl=60)
def cargar_datos(nombre_archivo):
    try:
        if nombre_archivo.endswith(".csv"):
            df = pd.read_csv(nombre_archivo)
        elif nombre_archivo.endswith(".xlsx"):
            df = pd.read_excel(nombre_archivo, engine="openpyxl", converters={'Peso': lambda x: float(str(x).replace(",", ".")) if isinstance(x, str) else x})  # Maneja las comas en "Peso"
        else:
            st.error("Formato de archivo no soportado. Por favor, sube un archivo .csv o .xlsx")
            return None
        # Convierte las columnas de fecha a datetime si existen
        if 'MONTH' in df.columns and 'DAY' in df.columns and 'YEAR' in df.columns:
             df['Fecha'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY ']], format='%Y%m%d', errors='coerce')

        return df

    except FileNotFoundError:
        st.error(f"Archivo {nombre_archivo} no encontrado.")
        return None
    except Exception as e:  # Captura otros errores durante la lectura
        st.error(f"Error al cargar el archivo: {e}")
        return None



# Subpágina para editar DataFrames
def editar_dataframe(df, nombre_archivo):
    st.subheader(f"Editar {nombre_archivo}")
    edited_df = st.data_editor(df)
    if st.button("Guardar cambios"):
        try:
            if nombre_archivo.endswith(".csv"):
                edited_df.to_csv(nombre_archivo, index=False)
            elif nombre_archivo.endswith(".xlsx"):
                edited_df.to_excel(nombre_archivo, index=False)  #  Considera usar openpyxl o xlsxwriter
            st.success("Cambios guardados con éxito!")
        except Exception as e:
            st.error(f"Error al guardar: {e}")

# Página principal
def main():
    st.title("Dashboard de Bases de Datos")

    # Cargar archivos
    archivos = [f for f in os.listdir(".") if f.endswith((".csv", ".xlsx"))]
    archivo_seleccionado = st.selectbox("Selecciona un archivo:", archivos)

    if archivo_seleccionado:
        df = cargar_datos(archivo_seleccionado)

        if df is not None:
            # Mostrar parámetros importantes
            col1, col2, col3 = st.columns(3)
            col1.metric("Número de filas", df.shape[0])
            col2.metric("Número de columnas", df.shape[1])
            col3.metric("Valores nulos", df.isna().sum().sum())

            # Mostrar el DataFrame
            st.dataframe(df, use_container_width=True)

            # Descripción de columnas (opcional)
            if st.checkbox("Mostrar descripción de columnas"):
                st.write(df.describe())

            # Editar DataFrame
            if st.button("Editar DataFrame"):
                editar_dataframe(df, archivo_seleccionado)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
