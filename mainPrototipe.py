import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib 

st.set_page_config(layout="wide", page_title="Prototipo analítico de perfiles de agresores")

RUTADATA_LIMPIA = r"C:\Users\ann_y\Downloads\datathonITM\P_agresora_Tlaxcala_limpio.xlsx" 

@st.cache_data(show_spinner="cargando datos")
def load_data(path_limpia):
    df_casos = pd.DataFrame()
    df_stats = pd.DataFrame()
    
    
    try:
        df_casos = pd.read_excel(path_limpia, sheet_name="Sheet1") 
    except FileNotFoundError:
        st.sidebar.error(f"no se encontró archivo en {path_limpia}")
    except Exception as e:
        st.sidebar.error(f" no lectura: {e}")

    if not df_casos.empty:
        for col in ['fisica', 'sexual', 'feminicida', 'agresores_consume_drogas']:
            if col in df_casos.columns:
                df_casos[col] = df_casos[col].astype(str).str.lower().str.strip()
        df_casos['agresores_edad'] = pd.to_numeric(df_casos['agresores_edad'], errors='coerce')

    return df_casos, df_stats


casos, stats = load_data(RUTADATA_LIMPIA) 


@st.cache_resource(show_spinner="entrenamiento")
def build_and_train_model(df):
    if df.empty or 'fisica' not in df.columns or 'agresores_edad' not in df.columns: return None
    df['Alto_Riesgo'] = np.where((df['fisica'] == 'si') | (df['sexual'] == 'si') | (df['feminicida'] == 'si'), 1, 0)
    features = ['agresores_edad', 'agresores_vinculo', 'agresores_consume_drogas', 'hecho_localidad', 'hecho_lugar_hechos']
    target = 'Alto_Riesgo'
    df_modelo = df.dropna(subset=['agresores_edad', 'agresores_vinculo'], how='any').copy()
    X = df_modelo[features]
    y = df_modelo[target]
    if len(X) < 100 or y.sum() == 0 or (len(y) - y.sum() == 0):
        st.sidebar.error("entrebnamiento fallido: no hay datos")
        return None
        
    categorical_features = ['agresores_vinculo', 'agresores_consume_drogas', 'hecho_localidad', 'hecho_lugar_hechos']
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    numerical_features = ['agresores_edad']
    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)], remainder='drop')
    ml_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))]) 
    ml_pipeline.fit(X, y)
    return ml_pipeline

modelo_prediccion = build_and_train_model(casos.copy()) if not casos.empty else None



def seccion_resumen_general(df_casos):
    """Muestra la distribución de las 5 features usadas por el modelo de ML."""
    st.title("Análisis Descriptivo de la Población Agresora")
    st.markdown("---")

    if df_casos.empty:
        st.error("no hay datos para el análisis.")
        return

 
    total_casos = len(df_casos)
    st.metric(label="Total de Registros de Casos", value=f"{total_casos:,}")
    st.markdown("---")
    
    st.subheader("Distribución por Características del Agresor")
    
    col1, col2 = st.columns(2)
    
    df_vinculo = df_casos['agresores_vinculo'].value_counts(normalize=True).mul(100).reset_index()
    df_vinculo.columns = ['Vínculo', 'Porcentaje']
    fig_vinculo = px.bar(
        df_vinculo.head(10).sort_values(by='Porcentaje', ascending=True), 
        x='Porcentaje', 
        y='Vínculo', 
        orientation='h',
        title='Principales Vínculos con la Víctima',
        color='Porcentaje',
        color_continuous_scale=px.colors.sequential.Teal
    )
    col1.plotly_chart(fig_vinculo, use_container_width=True)
    
    df_drogas = df_casos['agresores_consume_drogas'].str.capitalize().value_counts(normalize=True).mul(100).reset_index()
    df_drogas.columns = ['Consumo de Drogas', 'Porcentaje']
    fig_drogas = px.pie(
        df_drogas,
        values='Porcentaje',
        names='Consumo de Drogas',
        title='Consumo de Drogas del Agresor',
        hole=.3,
        color_discrete_sequence=px.colors.qualitative.T10
    )
    col2.plotly_chart(fig_drogas, use_container_width=True)

    st.markdown("---")

    st.subheader("Distribución de Edad del Agresor")
    age_data = df_casos['agresores_edad'].dropna()
    fig_edad = px.histogram(
        age_data,
        x='agresores_edad',
        nbins=30,
        title=f'Histograma de Edad del Agresor (Media: {age_data.mean():.1f})'
    )
    fig_edad.update_traces(marker_color='#56A0D3')
    st.plotly_chart(fig_edad, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Distribución Contextual de los Hechos")
    
    col3, col4 = st.columns(2)

    df_lugar = df_casos['hecho_lugar_hechos'].value_counts(normalize=True).mul(100).reset_index()
    df_lugar.columns = ['Lugar del Hecho', 'Porcentaje']
    fig_lugar = px.bar(
        df_lugar.head(8).sort_values(by='Porcentaje', ascending=True),
        x='Porcentaje', 
        y='Lugar del Hecho', 
        orientation='h',
        title='Principales Lugares de Ocurrencia',
        color_continuous_scale=px.colors.sequential.Plasma
    )
    col3.plotly_chart(fig_lugar, use_container_width=True)
    
    df_localidad = df_casos['hecho_localidad'].value_counts(normalize=True).mul(100).reset_index()
    df_localidad.columns = ['Localidad', 'Porcentaje']
    fig_localidad = px.bar(
        df_localidad.head(8).sort_values(by='Porcentaje', ascending=True),
        x='Porcentaje', 
        y='Localidad', 
        orientation='h',
        title='Principales Localidades de Ocurrencia',
        color='Porcentaje',
        color_continuous_scale=px.colors.sequential.Inferno
    )
    col4.plotly_chart(fig_localidad, use_container_width=True)

def seccion_prediccion_agresor(df_casos, model):
    st.title("Predicción de Posible Agresor")
    st.markdown("---")
    if model is None:
        st.error("modelo no está disponible")
        return
        
    st.markdown("Complete los campos para predecir el nivel de riesgo del agresor")
    st.subheader("Entrada para la predicción")
    vinculos = df_casos['agresores_vinculo'].dropna().unique()
    lugares_hechos = df_casos['hecho_lugar_hechos'].dropna().unique()
    localidades = df_casos['hecho_localidad'].dropna().unique()
    min_age = int(df_casos['agresores_edad'].min() if not df_casos['agresores_edad'].empty and df_casos['agresores_edad'].min() >= 18 else 18)
    max_age = int(df_casos['agresores_edad'].max() if not df_casos['agresores_edad'].empty else 70)
    default_age = 35

    with st.form("form_prediccion"):
        st.markdown("#### Variables del Caso")
        edad_agresor = st.slider("1. Edad del Agresor", min_age, max_age, default_age)
        col1, col2 = st.columns(2)
        vinculo = col1.selectbox("2. Parentesco con la Víctima", vinculos)
        drogas = col2.selectbox("3. Consumo de Drogas del Agresor", ['si', 'no', 'missing', 'no aplica'])
        col3, col4 = st.columns(2)
        localidad = col3.selectbox("4. Localidad de los Hechos", localidades)
        lugar_hechos = col4.selectbox("5. Lugar de los Hechos", lugares_hechos)
        submitted = st.form_submit_button("Obtener Predicción de Riesgo")
        
    if submitted:
        st.subheader("Resultados Predicción de Riesgo")
        input_data = pd.DataFrame({'agresores_edad': [edad_agresor], 'agresores_vinculo': [vinculo], 'agresores_consume_drogas': [drogas], 'hecho_localidad': [localidad], 'hecho_lugar_hechos': [lugar_hechos]})
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        riesgo = "ALTO" if prediction == 1 else "BAJO"
        prob_riesgo = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        st.metric(label="Nivel de Riesgo Predicho", value=riesgo)
        st.metric(label="Confianza del Modelo", value=f"{prob_riesgo:.2%}")
        if riesgo == "ALTO": st.error(f"El modelo clasifica el caso como **Riesgo ALTO** de presentar violencia grave.")
        else: st.success(f"El modelo clasifica el caso como **Riesgo BAJO** de presentar violencia grave.")
        st.info("Navega a la sección 'Soluciones por Perfil' para ver las recomendaciones basadas en este riesgo.")

def seccion_posibles_soluciones():
    st.title("Posibles Soluciones y Recomendaciones")
    st.markdown("---")
    st.markdown("""Rrecomendaciones de política pública, intervención legal y apoyo psicológico basadas en el perfil y el nivel de riesgo predicho del agresor.""")
    perfil_agresor = st.selectbox("Selecciona el Nivel de Riesgo Predicho:",['Riesgo ALTO', 'Riesgo BAJO'])
    st.subheader(f"Recomendaciones para: **{perfil_agresor}**")
    if 'ALTO' in perfil_agresor:
        st.error(" **Intervención de ALTO RIESGO**")
        st.markdown("""* **Legal/Policial:** Solicitud inmediata de **Orden de Protección de Emergencia**, asignación de patrulla de vigilancia cercana y uso de brazalete/monitoreo si es viable. * **Psicológica:** Terapia de control de ira y programas **obligatorios** de reeducación para agresores con historial de violencia de género. * **Social:** Alojamiento inmediato de la víctima en casa de refugio y contención del agresor.""")
    else:
        st.info("🔹 **Intervención de RIESGO BAJO**")
        st.markdown("""* **Legal/Policial:** Canalización a programas de **mediación y conciliación familiar**. Seguimiento social de control. * **Psicológica:** Asesoría psicoemocional para la víctima y el agresor, enfocada en la prevención de la escalada. * **Social:** Información sobre derechos y recursos comunitarios existentes para prevenir la violencia.""")

def main():
    
    st.sidebar.title("Prototipo analítico de perfiles de agresores")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Diagnóstico de Datos")
    
    if casos.empty:
        st.title("no hubo carga de datos")
        st.markdown("error en la carga de datos, revisar ruta")
        return 

    if modelo_prediccion is None:
        st.sidebar.error("rf no entrenado.")
        st.sidebar.info("entrenamiento falló")
    else:
        st.sidebar.success("Prototipo listo")
        

    seccion = st.sidebar.radio(
        "Selecciona una sección:",
        ["Resumen General", "Predicción de Agresor", "Soluciones por Perfil"] 
    )

    if seccion == "Resumen General":
        seccion_resumen_general(casos) 
    elif seccion == "Predicción de Agresor":
        seccion_prediccion_agresor(casos, modelo_prediccion) 
    elif seccion == "Soluciones por Perfil":
        seccion_posibles_soluciones()

if __name__ == "__main__":
    main()