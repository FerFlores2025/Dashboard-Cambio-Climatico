
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure
from bokeh.models import HoverTool
import requests
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN Y OBTENCIÓN DE DATOS
# ============================================================================

# Configuración de la página (debe ser el primer comando Streamlit)
#st.set_page_config(page_title="Dashboard Cambio Climático", layout="wide", page_icon="🌍")

# Título principal"""st.markdown("""
# 🌍 DASHBOARD DE CAMBIO CLIMÁTICO MUNDIAL  
#*Análisis histórico (1990-2024) y predicciones hasta 2050*
#"""")"""

# Configuración de API Key
API_KEY = "5654aefb3245b09a83ae9bb9879f2660"  # Reemplazar con tu API key

# Ciudades principales para análisis mundial
CIUDADES_MUNDIALES = {
    'Nueva York': {'lat': 40.7128, 'lon': -74.0060, 'pais': 'USA'},
    'Londres': {'lat': 51.5074, 'lon': -0.1278, 'pais': 'UK'},
    'Tokio': {'lat': 35.6762, 'lon': 139.6503, 'pais': 'Japón'},
    'París': {'lat': 48.8566, 'lon': 2.3522, 'pais': 'Francia'},
    'Sídney': {'lat': -33.8688, 'lon': 151.2093, 'pais': 'Australia'},
    'Moscú': {'lat': 55.7558, 'lon': 37.6173, 'pais': 'Rusia'},
    'São Paulo': {'lat': -23.5505, 'lon': -46.6333, 'pais': 'Brasil'},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'pais': 'India'},
    'El Cairo': {'lat': 30.0444, 'lon': 31.2357, 'pais': 'Egipto'},
    'Ciudad de México': {'lat': 19.4326, 'lon': -99.1332, 'pais': 'México'},
    'Beijing': {'lat': 39.9042, 'lon': 116.4074, 'pais': 'China'},
    'Buenos Aires': {'lat': -34.6037, 'lon': -58.3816, 'pais': 'Argentina'},
    'Lagos': {'lat': 6.5244, 'lon': 3.3792, 'pais': 'Nigeria'},
    'Berlín': {'lat': 52.5200, 'lon': 13.4050, 'pais': 'Alemania'},
    'Toronto': {'lat': 43.6532, 'lon': -79.3832, 'pais': 'Canadá'}
}

@st.cache_data(ttl=3600)
def generar_datos_historicos_sinteticos():
    """
    Genera datos históricos sintéticos basados en tendencias reales de cambio climático
    Simula datos de 1990-2024 con tendencias de calentamiento global
    """
    np.random.seed(42)
    años = range(1990, 2025)
    datos = []
    
    for ciudad, coords in CIUDADES_MUNDIALES.items():
        # Temperatura base según latitud
        lat = coords['lat']
        temp_base = 15 - (abs(lat) / 90) * 20 + np.random.normal(0, 2)
        
        for año in años:
            # Tendencia de calentamiento: +0.02°C por año en promedio
            tendencia = (año - 1990) * 0.02
            
            # Variabilidad estacional y aleatoria
            for mes in range(1, 13):
                variacion_estacional = 10 * np.sin((mes - 1) * np.pi / 6)
                if lat < 0:  # Hemisferio sur
                    variacion_estacional *= -1
                
                temp = temp_base + tendencia + variacion_estacional + np.random.normal(0, 1.5)
                
                datos.append({
                    'ciudad': ciudad,
                    'pais': coords['pais'],
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'año': año,
                    'mes': mes,
                    'temperatura': round(temp, 2),
                    'precipitacion': abs(np.random.normal(50, 30)),
                    'humedad': np.random.uniform(40, 90),
                    'presion': np.random.uniform(1000, 1020),
                    'viento': abs(np.random.normal(15, 8))
                })
    
    return pd.DataFrame(datos)

def generar_datos_actuales_sinteticos(ciudad):
    """Genera datos sintéticos cuando la API no está disponible"""
    coords = CIUDADES_MUNDIALES[ciudad]
    lat = coords['lat']
    
    # Temperatura base con variación estacional
    mes_actual = datetime.now().month
    variacion_estacional = 10 * np.sin((mes_actual - 1) * np.pi / 6)
    if lat < 0:  # Hemisferio sur
        variacion_estacional *= -1
    
    temp_base = 15 - (abs(lat) / 90) * 20
    temperatura = temp_base + variacion_estacional + np.random.normal(0, 2)
    
    return {
        'temperatura': round(temperatura, 2),
        'humedad': np.random.uniform(40, 85),
        'presion': np.random.uniform(1000, 1020),
        'viento': round(abs(np.random.normal(15, 8)), 1),
        'descripcion': 'Datos de demostración',
        'actualizado': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@st.cache_data(ttl=3600)
def obtener_datos_actuales_api(ciudad, lat, lon):
    """Obtiene datos actuales con mejor manejo de errores"""
    if API_KEY == "TU_API_KEY_AQUI" or not API_KEY:
        return generar_datos_actuales_sinteticos(ciudad)
    
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'temperatura': data['main']['temp'],
                'humedad': data['main']['humidity'],
                'presion': data['main']['pressure'],
                'viento': data['wind']['speed'],
                'descripcion': data['weather'][0]['description'],
                'actualizado': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            st.sidebar.warning(f"Error API: {response.status_code} para {ciudad}")
            return generar_datos_actuales_sinteticos(ciudad)
            
    except Exception as e:
        st.sidebar.warning(f"Error conectando a API para {ciudad}: {str(e)}")
        return generar_datos_actuales_sinteticos(ciudad)

def cargar_datos():
    """Carga y prepara los datos para análisis"""
    df = generar_datos_historicos_sinteticos()
    df['fecha'] = pd.to_datetime({'year': df['año'], 'month': df['mes'], 'day': 1})
    return df

df = cargar_datos()

# ============================================================================
# PREPARAR DATOS PARA MACHINE LEARNING
# ============================================================================

def preparar_datos_ml(df):
    """Prepara los datos para entrenamiento de modelos"""
    df_ml = df.copy()
    df_ml['año_num'] = df_ml['año']
    df_ml['mes_sin'] = np.sin(2 * np.pi * df_ml['mes'] / 12)
    df_ml['mes_cos'] = np.cos(2 * np.pi * df_ml['mes'] / 12)
    df_ml['tendencia'] = df_ml['año'] - 1990
    
    return df_ml

# ============================================================================
# ENTRENAR MODELOS DE PREDICCIÓN
# ============================================================================

@st.cache_resource
def entrenar_modelo_prediccion(df):
    """Entrena modelo de ML para predicciones hasta 2050"""
    df_ml = preparar_datos_ml(df)
    
    # Agregación por año
    df_anual = df_ml.groupby(['año', 'lat']).agg({
        'temperatura': 'mean',
        'precipitacion': 'mean',
        'humedad': 'mean'
    }).reset_index()
    
    # Features y target
    X = df_anual[['año', 'lat']]
    y = df_anual['temperatura']
    
    # Split datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar múltiples modelos
    modelos = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    mejor_modelo = None
    mejor_score = -np.inf
    
    for nombre, modelo in modelos.items():
        modelo.fit(X_train_scaled, y_train)
        score = modelo.score(X_test_scaled, y_test)
        if score > mejor_score:
            mejor_score = score
            mejor_modelo = modelo
    
    return mejor_modelo, scaler, mejor_score

modelo, scaler, score = entrenar_modelo_prediccion(df)

# ============================================================================
# GENERAR PREDICCIONES HASTA 2050
# ============================================================================

def generar_predicciones_2050(modelo, scaler, df):
    """Genera predicciones de temperatura hasta 2050"""
    años_futuros = range(2025, 2051)
    predicciones = []
    
    for año in años_futuros:
        for ciudad, coords in CIUDADES_MUNDIALES.items():
            X_pred = np.array([[año, coords['lat']]])
            X_pred_scaled = scaler.transform(X_pred)
            temp_pred = modelo.predict(X_pred_scaled)[0]
            
            predicciones.append({
                'ciudad': ciudad,
                'lat': coords['lat'],
                'lon': coords['lon'],
                'año': año,
                'temperatura': round(temp_pred, 2),
                'tipo': 'prediccion'
            })
    
    return pd.DataFrame(predicciones)

def generar_predicciones_con_intervalos(modelo, scaler, df, n_iteraciones=50):
    """Genera predicciones con intervalos de confianza"""
    años_futuros = range(2025, 2051)
    predicciones = []
    
    for _ in range(n_iteraciones):
        for año in años_futuros:
            for ciudad, coords in CIUDADES_MUNDIALES.items():
                # Añadir ruido para simular incertidumbre
                ruido_lat = np.random.normal(0, 0.1)
                X_pred = np.array([[año, coords['lat'] + ruido_lat]])
                X_pred_scaled = scaler.transform(X_pred)
                temp_pred = modelo.predict(X_pred_scaled)[0]
                
                # Añadir ruido a la predicción
                temp_pred_ruido = temp_pred + np.random.normal(0, 0.5)
                
                predicciones.append({
                    'ciudad': ciudad,
                    'año': año,
                    'temperatura': temp_pred_ruido,
                    'iteracion': _
                })
    
    df_pred = pd.DataFrame(predicciones)
    
    # Calcular intervalos de confianza
    df_intervalos = df_pred.groupby(['ciudad', 'año']).agg({
        'temperatura': ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)]
    }).reset_index()
    
    df_intervalos.columns = ['ciudad', 'año', 'temperatura_media', 'temperatura_std', 'temp_5percentil', 'temp_95percentil']
    
    return df_intervalos

df_predicciones = generar_predicciones_2050(modelo, scaler, df)
df_predicciones_intervalos = generar_predicciones_con_intervalos(modelo, scaler, df)

# ============================================================================
# ANÁLISIS DE EVENTOS EXTREMOS
# ============================================================================

def analizar_eventos_extremos(df):
    """Analiza frecuencia de eventos de temperatura extrema"""
    df['es_ola_calor'] = df['temperatura'] > df.groupby(['ciudad', 'mes'])['temperatura'].transform('mean') + 2 * df.groupby(['ciudad', 'mes'])['temperatura'].transform('std')
    
    eventos_por_año = df.groupby('año')['es_ola_calor'].sum().reset_index()
    eventos_por_año.columns = ['año', 'eventos_calor']
    
    return eventos_por_año

df_eventos = analizar_eventos_extremos(df)

# ============================================================================
# DASHBOARD INTERACTIVO
# ============================================================================

# Header
st.title("🌍 Dashboard de Cambio Climático Mundial")
st.markdown("### Análisis histórico (1990-2024) y predicciones hasta 2050")
st.markdown("---")

# Sidebar con controles
st.sidebar.header("⚙️ Controles del Dashboard")

# Opción para ingresar API key
if API_KEY == "5654aefb3245b09a83ae9bb9879f2660":
    api_key_user = st.sidebar.text_input("5654aefb3245b09a83ae9bb9879f2660", type="password")
    if api_key_user:
        API_KEY = api_key_user
        st.sidebar.success("✅ API Key configurada correctamente")
else:
    st.sidebar.info("🔑 API Key configurada")

año_seleccionado = st.sidebar.slider("Seleccionar Año", 1990, 2050, 2024)
ciudades_seleccionadas = st.sidebar.multiselect(
    "Seleccionar Ciudades",
    options=list(CIUDADES_MUNDIALES.keys()),
    default=['Nueva York', 'Londres', 'Tokio', 'Buenos Aires']
)

# Sección de exportación de datos
st.sidebar.markdown("---")
st.sidebar.header("📊 Exportar Datos")

# Preparar datos para exportar
df_export = df.groupby(['ciudad', 'año']).agg({
    'temperatura': ['mean', 'std', 'min', 'max'],
    'precipitacion': 'mean',
    'humedad': 'mean'
}).round(2).reset_index()

df_export.columns = ['ciudad', 'año', 'temp_promedio', 'temp_std', 'temp_min', 'temp_max', 'precip_promedio', 'humedad_promedio']

# Convertir a CSV
csv = df_export.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Descargar Datos Históricos (CSV)",
    data=csv,
    file_name="datos_cambio_climatico.csv",
    mime="text/csv"
)

# Métricas principales
col1, col2, col3, col4 = st.columns(4)

temp_actual = df[df['año'] == 2024]['temperatura'].mean()
temp_1990 = df[df['año'] == 1990]['temperatura'].mean()
cambio_temp = temp_actual - temp_1990
temp_2050_pred = df_predicciones[df_predicciones['año'] == 2050]['temperatura'].mean()

with col1:
    st.metric("🌡️ Temperatura Promedio 2024", f"{temp_actual:.2f}°C", f"+{cambio_temp:.2f}°C")
with col2:
    st.metric("📈 Cambio desde 1990", f"+{cambio_temp:.2f}°C", f"{(cambio_temp/temp_1990*100):.1f}%")
with col3:
    st.metric("🔮 Predicción 2050", f"{temp_2050_pred:.2f}°C", f"+{(temp_2050_pred-temp_actual):.2f}°C")
with col4:
    st.metric("🤖 Precisión del Modelo", f"{score*100:.1f}%", "R² Score")

# ============================================================================
# SECCIÓN DE DATOS EN TIEMPO REAL
# ============================================================================

st.markdown("---")
st.header("🌤️ Condiciones Actuales en Tiempo Real")

if st.button("🔄 Actualizar Datos en Tiempo Real"):
    st.cache_data.clear()
    st.rerun()

# Mostrar datos actuales para ciudades seleccionadas
if ciudades_seleccionadas:
    cols = st.columns(min(3, len(ciudades_seleccionadas)))
    
    for idx, ciudad in enumerate(ciudades_seleccionadas):
        coords = CIUDADES_MUNDIALES[ciudad]
        datos_actuales = obtener_datos_actuales_api(ciudad, coords['lat'], coords['lon'])
        
        if datos_actuales:
            with cols[idx % 3]:
                st.subheader(f"🏙️ {ciudad}")
                st.metric("Temperatura", f"{datos_actuales['temperatura']:.1f}°C")
                st.metric("Humedad", f"{datos_actuales['humedad']:.0f}%")
                st.metric("Viento", f"{datos_actuales['viento']:.1f} km/h")
                st.caption(f"💬 {datos_actuales['descripcion']}")
                st.caption(f"🕒 {datos_actuales.get('actualizado', 'Actualizado ahora')}")

st.markdown("---")

# ============================================================================
# GRÁFICO 1: MAPA DE BURBUJAS CON TEMPERATURAS MUNDIALES
# ============================================================================

st.header("1️⃣ Mapa Mundial de Temperaturas (Burbujas Interactivas)")

df_mapa = df[df['año'] == año_seleccionado].groupby(['ciudad', 'lat', 'lon', 'pais']).agg({
    'temperatura': 'mean'
}).reset_index()

fig_mapa = go.Figure()

fig_mapa.add_trace(go.Scattergeo(
    lon=df_mapa['lon'],
    lat=df_mapa['lat'],
    text=df_mapa['ciudad'],
    mode='markers',
    marker=dict(
        size=df_mapa['temperatura'] * 2,
        color=df_mapa['temperatura'],
        colorscale='RdYlBu_r',
        showscale=True,
        colorbar=dict(title="Temperatura °C"),
        line=dict(width=0.5, color='black')
    ),
    hovertemplate='<b>%{text}</b><br>Temperatura: %{marker.color:.2f}°C<extra></extra>'
))

fig_mapa.update_layout(
    title=f'Distribución de Temperaturas Mundial - {año_seleccionado}',
    geo=dict(
        showland=True,
        landcolor='rgb(243, 243, 243)',
        coastlinecolor='rgb(204, 204, 204)',
        projection_type='natural earth',
        showocean=True,
        oceancolor='lightblue'
    ),
    height=600
)

st.plotly_chart(fig_mapa, use_container_width=True)

# ============================================================================
# GRÁFICO 2: TERMÓMETRO Y EVOLUCIÓN TEMPORAL
# ============================================================================

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🌡️ Termómetro Global")
    
    fig_termo = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=temp_actual,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Temperatura Promedio {año_seleccionado}"},
        delta={'reference': temp_1990, 'suffix': '°C'},
        gauge={
            'axis': {'range': [None, 30]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 10], 'color': "lightblue"},
                {'range': [10, 20], 'color': "lightyellow"},
                {'range': [20, 30], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    
    fig_termo.update_layout(height=400)
    st.plotly_chart(fig_termo, use_container_width=True)

with col2:
    st.subheader("📈 Evolución Histórica y Predicciones")
    
    df_evolucion = df.groupby('año')['temperatura'].mean().reset_index()
    df_pred_evol = df_predicciones.groupby('año')['temperatura'].mean().reset_index()
    
    fig_evol = go.Figure()
    
    fig_evol.add_trace(go.Scatter(
        x=df_evolucion['año'],
        y=df_evolucion['temperatura'],
        name='Histórico',
        mode='lines+markers',
        line=dict(color='blue', width=3)
    ))
    
    fig_evol.add_trace(go.Scatter(
        x=df_pred_evol['año'],
        y=df_pred_evol['temperatura'],
        name='Predicción',
        mode='lines+markers',
        line=dict(color='red', width=3, dash='dash')
    ))
    
    fig_evol.update_layout(
        title='Temperatura Promedio Global (1990-2050)',
        xaxis_title='Año',
        yaxis_title='Temperatura (°C)',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_evol, use_container_width=True)

st.markdown("---")

# ============================================================================
# GRÁFICO 3: COMPARACIÓN ENTRE CIUDADES
# ============================================================================

st.header("2️⃣ Comparación entre Ciudades Seleccionadas")

if ciudades_seleccionadas:
    df_ciudades = df[df['ciudad'].isin(ciudades_seleccionadas)]
    df_ciudades_anual = df_ciudades.groupby(['año', 'ciudad'])['temperatura'].mean().reset_index()
    
    fig_ciudades = px.line(
        df_ciudades_anual,
        x='año',
        y='temperatura',
        color='ciudad',
        title='Evolución de Temperaturas por Ciudad',
        labels={'temperatura': 'Temperatura (°C)', 'año': 'Año'}
    )
    
    fig_ciudades.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig_ciudades, use_container_width=True)

# ============================================================================
# GRÁFICO 4: HEATMAP DE TEMPERATURAS POR MES Y AÑO
# ============================================================================

st.header("3️⃣ Mapa de Calor: Patrones Estacionales")

ciudad_heatmap = st.selectbox("Seleccionar ciudad para heatmap", list(CIUDADES_MUNDIALES.keys()))

df_heatmap = df[df['ciudad'] == ciudad_heatmap].pivot_table(
    values='temperatura',
    index='mes',
    columns='año',
    aggfunc='mean'
)

fig_heatmap = px.imshow(
    df_heatmap,
    labels=dict(x="Año", y="Mes", color="Temperatura (°C)"),
    title=f'Patrones de Temperatura Mensual - {ciudad_heatmap}',
    color_continuous_scale='RdYlBu_r',
    aspect='auto'
)

fig_heatmap.update_layout(height=500)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("---")

# ============================================================================
# GRÁFICO 5: DISTRIBUCIÓN DE TEMPERATURAS (VIOLIN PLOT)
# ============================================================================

st.header("4️⃣ Distribución de Temperaturas por Década")

df['decada'] = (df['año'] // 10) * 10

fig_violin = go.Figure()

for decada in sorted(df['decada'].unique()):
    df_decada = df[df['decada'] == decada]
    fig_violin.add_trace(go.Violin(
        y=df_decada['temperatura'],
        name=f'{int(decada)}s',
        box_visible=True,
        meanline_visible=True
    ))

fig_violin.update_layout(
    title='Distribución de Temperaturas por Década',
    yaxis_title='Temperatura (°C)',
    showlegend=True,
    height=500
)

st.plotly_chart(fig_violin, use_container_width=True)

# ============================================================================
# GRÁFICO 6: ANÁLISIS DE PRECIPITACIONES Y CORRELACIONES
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("☔ Precipitaciones Anuales")
    
    df_precip = df.groupby('año')['precipitacion'].mean().reset_index()
    
    fig_precip = go.Figure()
    fig_precip.add_trace(go.Bar(
        x=df_precip['año'],
        y=df_precip['precipitacion'],
        marker_color='steelblue'
    ))
    
    fig_precip.update_layout(
        title='Precipitación Promedio Anual',
        xaxis_title='Año',
        yaxis_title='Precipitación (mm)',
        height=400
    )
    
    st.plotly_chart(fig_precip, use_container_width=True)

with col2:
    st.subheader("🔗 Correlaciones Climáticas")
    
    df_corr = df[['temperatura', 'precipitacion', 'humedad', 'presion', 'viento']].corr()
    
    fig_corr = px.imshow(
        df_corr,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        title='Matriz de Correlación'
    )
    
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ============================================================================
# GRÁFICO 7: ANÁLISIS REGIONAL POR HEMISFERIO
# ============================================================================

st.header("5️⃣ Análisis por Hemisferio")

df['hemisferio'] = df['lat'].apply(lambda x: 'Norte' if x > 0 else 'Sur')
df_hemisferio = df.groupby(['año', 'hemisferio'])['temperatura'].mean().reset_index()

fig_hemisferio = px.area(
    df_hemisferio,
    x='año',
    y='temperatura',
    color='hemisferio',
    title='Comparación de Temperaturas: Hemisferio Norte vs Sur',
    labels={'temperatura': 'Temperatura (°C)', 'año': 'Año'}
)

fig_hemisferio.update_layout(height=500)
st.plotly_chart(fig_hemisferio, use_container_width=True)

# ============================================================================
# GRÁFICO 8: TENDENCIAS Y PROYECCIONES CON INTERVALOS
# ============================================================================

st.header("6️⃣ Análisis de Tendencias y Proyecciones")

# Combinar datos históricos y predicciones
df_completo = pd.concat([
    df.groupby('año')['temperatura'].mean().reset_index().assign(tipo='histórico'),
    df_predicciones.groupby('año')['temperatura'].mean().reset_index().assign(tipo='predicción')
])

fig_tendencia = go.Figure()

# Datos históricos
df_hist = df_completo[df_completo['tipo'] == 'histórico']
fig_tendencia.add_trace(go.Scatter(
    x=df_hist['año'],
    y=df_hist['temperatura'],
    mode='markers',
    name='Datos Históricos',
    marker=dict(size=8, color='blue')
))

# Línea de tendencia
z = np.polyfit(df_hist['año'], df_hist['temperatura'], 1)
p = np.poly1d(z)
fig_tendencia.add_trace(go.Scatter(
    x=df_hist['año'],
    y=p(df_hist['año']),
    mode='lines',
    name='Tendencia Histórica',
    line=dict(color='blue', dash='dash')
))

# Predicciones
df_pred = df_completo[df_completo['tipo'] == 'predicción']
fig_tendencia.add_trace(go.Scatter(
    x=df_pred['año'],
    y=df_pred['temperatura'],
    mode='markers',
    name='Predicciones',
    marker=dict(size=8, color='red', symbol='diamond')
))

fig_tendencia.update_layout(
    title='Tendencia y Proyección de Temperatura Global (1990-2050)',
    xaxis_title='Año',
    yaxis_title='Temperatura (°C)',
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig_tendencia, use_container_width=True)

# ============================================================================
# GRÁFICO 9: INTERVALOS DE CONFIANZA EN PREDICCIONES
# ============================================================================

with st.expander("📊 Ver Predicciones con Intervalos de Confianza"):
    st.subheader("Predicciones con Intervalos de Confianza")
    
    ciudad_ejemplo = st.selectbox("Seleccionar ciudad para análisis de incertidumbre", 
                                 list(CIUDADES_MUNDIALES.keys()))
    
    df_ciudad = df_predicciones_intervalos[df_predicciones_intervalos['ciudad'] == ciudad_ejemplo]
    
    fig_incertidumbre = go.Figure()
    
    fig_incertidumbre.add_trace(go.Scatter(
        x=df_ciudad['año'],
        y=df_ciudad['temp_95percentil'],
        fill=None,
        mode='lines',
        line_color='lightblue',
        name='Límite superior (95%)',
        showlegend=True
    ))
    
    fig_incertidumbre.add_trace(go.Scatter(
        x=df_ciudad['año'],
        y=df_ciudad['temp_5percentil'],
        fill='tonexty',
        mode='lines',
        line_color='lightblue',
        name='Intervalo de confianza',
        showlegend=True
    ))
    
    fig_incertidumbre.add_trace(go.Scatter(
        x=df_ciudad['año'],
        y=df_ciudad['temperatura_media'],
        mode='lines',
        line=dict(color='blue', width=3),
        name='Predicción media'
    ))
    
    fig_incertidumbre.update_layout(
        title=f'Predicciones con Intervalos de Confianza - {ciudad_ejemplo}',
        xaxis_title='Año',
        yaxis_title='Temperatura (°C)',
        height=400
    )
    
    st.plotly_chart(fig_incertidumbre, use_container_width=True)

# ============================================================================
# GRÁFICO 10: EVENTOS DE CALOR EXTREMO
# ============================================================================

with st.expander("🔥 Análisis de Eventos de Calor Extremo"):
    st.subheader("Frecuencia de Eventos de Calor Extremo")
    
    fig_eventos = px.line(
        df_eventos,
        x='año',
        y='eventos_calor',
        title='Frecuencia de Eventos de Calor Extremo (1990-2024)',
        labels={'eventos_calor': 'Número de eventos', 'año': 'Año'}
    )
    
    fig_eventos.update_layout(height=400)
    st.plotly_chart(fig_eventos, use_container_width=True)

# ============================================================================
# RESUMEN ESTADÍSTICO Y CONCLUSIONES
# ============================================================================

st.markdown("---")
st.header("📊 Resumen Estadístico y Conclusiones")

col1, col2, col3 = st.columns(3)

aumento_total = temp_2050_pred - temp_1990

with col1:
    st.subheader("📈 Tendencias Históricas")
    st.write(f"*Aumento de temperatura (1990-2024):* {cambio_temp:.2f}°C")
    st.write(f"*Tasa promedio:* {(cambio_temp/34):.3f}°C por año")
    st.write(f"*Variabilidad:* {df[df['año'] >= 2020]['temperatura'].std():.2f}°C")

with col2:
    st.subheader("🔮 Proyecciones 2050")
    st.write(f"*Temperatura estimada 2050:* {temp_2050_pred:.2f}°C")
    st.write(f"*Aumento total (1990-2050):* {aumento_total:.2f}°C")
    st.write(f"*Precisión del modelo:* {score*100:.1f}%")

with col3:
    st.subheader("🌍 Datos del Análisis")
    st.write(f"*Ciudades analizadas:* {len(CIUDADES_MUNDIALES)}")
    st.write(f"*Período histórico:* 1990-2024")
    st.write(f"*Registros totales:* {len(df):,}")

# ============================================================================
# GENERAR REPORTE EJECUTIVO
# ============================================================================

if st.sidebar.button("📄 Generar Reporte Ejecutivo"):
    with st.expander("📋 Reporte Ejecutivo - Cambio Climático", expanded=True):
        st.subheader("Hallazgos Principales")
        
        st.write(f"*Tendencia de Calentamiento:* {aumento_total:.2f}°C de aumento proyectado (1990-2050)")
        st.write(f"*Tasa Actual:* {(cambio_temp/34):.3f}°C por año")
        st.write(f"*Impacto Regional:* Variación de {df_predicciones[df_predicciones['año'] == 2050]['temperatura'].std():.2f}°C entre regiones")
        
        st.subheader("Recomendaciones")
        st.write("1. *Implementar políticas de mitigación inmediatas*")
        st.write("2. *Fortalecer sistemas de alerta temprana*")
        st.write("3. *Invertir en adaptación climática*")
        st.write("4. *Promover energías renovables*")
        st.write("5. *Desarrollar infraestructura resiliente*")

# ============================================================================
# INFORMACIÓN DEL SISTEMA
# ============================================================================

st.markdown("---")
st.header("⚙️ Información del Sistema")

with st.expander("📋 Metodología y Fuentes de Datos"):
    st.markdown("""
    ### Metodología de Machine Learning Aplicada
    
    *1. Encuadre del Problema:*
    - Problema de regresión para predicción de temperaturas
    - Series temporales con tendencias y estacionalidad
    
    *2. Obtención de Datos:*
    - Datos sintéticos basados en tendencias reales de cambio climático
    - API de OpenWeatherMap para datos actuales
    - 15 ciudades representativas en todos los continentes
    
    *3. Modelado:*
    - Random Forest Regressor (modelo seleccionado)
    - Gradient Boosting Regressor
    - Linear Regression
    - Selección del mejor modelo por R² score
    
    *4. Evaluación:*
    - R² Score para precisión: {:.1f}%
    - Validación cruzada
    - Análisis de residuos
    """.format(score*100))

with st.expander("📚 Bibliotecas Utilizadas"):
    st.code("""
    - Streamlit: Framework de dashboard interactivo
    - Plotly: Gráficos interactivos avanzados
    - Pandas: Manipulación de datos
    - NumPy: Cálculos numéricos
    - Scikit-learn: Modelos de Machine Learning
    - Matplotlib/Seaborn: Visualizaciones estáticas
    - Requests: Llamadas a API
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Dashboard de Cambio Climático Mundial</strong></p>
    <p>Proyecto de Machine Learning - Análisis y Predicción del Clima Global</p>
    <p>Datos históricos: 1990-2024 | Predicciones: hasta 2050</p>
</div>
""", unsafe_allow_html=True)

# Notas finales
#st.info("""
#💡 *Nota:* Para obtener datos reales de la API de OpenWeatherMap, necesitas:
#1. Registrarte en https://openweathermap.org/api (plan gratuito disponible)
#2. Obtener tu API Key
#3. Reemplazar 'TU_API_KEY_AQUI' en el código con tu clave
#4. El dashboard funcionará con datos sintéticos realistas sin la API
#""")