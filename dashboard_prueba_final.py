
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import requests
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')



# CONFIGURACIÓN 

st.set_page_config(
    page_title=" Dashboard Cambio Climatico",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded"
)

# ESTILO 
st.markdown("""
<style>
    /* Variables de color profesionales */
    :root {
        --primary-blue: #2E86AB;
        --secondary-blue: #A23B72;
        --success-green: #06A77D;
        --warning-orange: #F18F01;
        --danger-red: #C73E1D;
        --neutral-gray: #5D6D7E;
        --light-bg: #F8F9FA;
    }
    
    /* Layout principal */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Headers con estilo ejecutivo */
    h1 {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1A202C !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.5px;
    }
    
    h2 {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        color: #2D3748 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        color: #4A5568 !important;
        margin-top: 1.5rem !important;
    }
    
    /* KPI Cards mejoradas */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid var(--primary-blue);
        transition: transform 0.2s, box-shadow 0.2s;
        height: 100%;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.12);
    }
    
    .kpi-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }
    
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2D3748;
        margin-bottom: 0.25rem;
    }
    
    .kpi-change {
        font-size: 0.875rem;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
    }
    
    .kpi-change.positive {
        background-color: #FED7D7;
        color: #C53030;
    }
    
    .kpi-change.negative {
        background-color: #C6F6D5;
        color: #22543D;
    }
    
    /* Métricas nativas de Streamlit  */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #2D3748 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #718096 !important;
        text-transform: uppercase !important;
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .alert-warning {
        background-color: #FFFAF0;
        border-color: #F18F01;
        color: #744210;
    }
    
    .alert-danger {
        background-color: #FFF5F5;
        border-color: #C73E1D;
        color: #742A2A;
    }
    
    .alert-info {
        background-color: #EBF8FF;
        border-color: #2E86AB;
        color: #2C5282;
    }
    
    /* Tabs  */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: white;
        border-radius: 8px;
        padding: 0.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 6px;
        padding: 0 1.5rem;
        font-weight: 500;
        color: #4A5568;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2E86AB 0%, #1E5F7E 100%);
        color: white !important;
    }
    
    /* Sidebar  */
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    .stSidebar [data-testid="stSidebarNav"] {
        background-color: transparent;
    }
    
    /* Botones  */
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB 0%, #1E5F7E 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(46, 134, 171, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(46, 134, 171, 0.4);
    }
    
    /* Info boxes */
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .insight-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
    }
    
    /* Animaciones suaves */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Tooltips  */
    .tooltip-icon {
        display: inline-block;
        width: 18px;
        height: 18px;
        background-color: #CBD5E0;
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 12px;
        font-weight: bold;
        margin-left: 5px;
        cursor: help;
    }
    
    /* Divisores elegantes */
    .divider {
        height: 1px;
        background: linear-gradient(to right, transparent, #E2E8F0, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# CONFIGURACIÓN Y DATOS 

API_KEY = "5654aefb3245b09a83ae9bb9879f2660"

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
    np.random.seed(42)
    años = range(1990, 2025)
    datos = []
    
    for ciudad, coords in CIUDADES_MUNDIALES.items():
        lat = coords['lat']
        temp_base = 15 - (abs(lat) / 90) * 20 + np.random.normal(0, 2)
        
        for año in años:
            tendencia = (año - 1990) * 0.02
            
            for mes in range(1, 13):
                variacion_estacional = 10 * np.sin((mes - 1) * np.pi / 6)
                if lat < 0:
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
    coords = CIUDADES_MUNDIALES[ciudad]
    lat = coords['lat']
    mes_actual = datetime.now().month
    variacion_estacional = 10 * np.sin((mes_actual - 1) * np.pi / 6)
    if lat < 0:
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
    if API_KEY == "5654aefb3245b09a83ae9bb9879f2660" or not API_KEY:
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
            return generar_datos_actuales_sinteticos(ciudad)
            
    except Exception as e:
        return generar_datos_actuales_sinteticos(ciudad)

def cargar_datos():
    df = generar_datos_historicos_sinteticos()
    df['fecha'] = pd.to_datetime({'year': df['año'], 'month': df['mes'], 'day': 1})
    return df

df = cargar_datos()

def preparar_datos_ml(df):
    df_ml = df.copy()
    df_ml['año_num'] = df_ml['año']
    df_ml['mes_sin'] = np.sin(2 * np.pi * df_ml['mes'] / 12)
    df_ml['mes_cos'] = np.cos(2 * np.pi * df_ml['mes'] / 12)
    df_ml['tendencia'] = df_ml['año'] - 1990
    return df_ml

@st.cache_resource
def entrenar_modelo_prediccion(df):
    df_ml = preparar_datos_ml(df)
    df_anual = df_ml.groupby(['año', 'lat']).agg({
        'temperatura': 'mean',
        'precipitacion': 'mean',
        'humedad': 'mean'
    }).reset_index()
    
    X = df_anual[['año', 'lat']]
    y = df_anual['temperatura']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
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

def generar_predicciones_2050(modelo, scaler, df):
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
    años_futuros = range(2025, 2051)
    predicciones = []
    
    for _ in range(n_iteraciones):
        for año in años_futuros:
            for ciudad, coords in CIUDADES_MUNDIALES.items():
                ruido_lat = np.random.normal(0, 0.1)
                X_pred = np.array([[año, coords['lat'] + ruido_lat]])
                X_pred_scaled = scaler.transform(X_pred)
                temp_pred = modelo.predict(X_pred_scaled)[0]
                temp_pred_ruido = temp_pred + np.random.normal(0, 0.5)
                
                predicciones.append({
                    'ciudad': ciudad,
                    'año': año,
                    'temperatura': temp_pred_ruido,
                    'iteracion': _
                })
    
    df_pred = pd.DataFrame(predicciones)
    df_intervalos = df_pred.groupby(['ciudad', 'año']).agg({
        'temperatura': ['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)]
    }).reset_index()
    
    df_intervalos.columns = ['ciudad', 'año', 'temperatura_media', 'temperatura_std', 'temp_5percentil', 'temp_95percentil']
    return df_intervalos

df_predicciones = generar_predicciones_2050(modelo, scaler, df)
df_predicciones_intervalos = generar_predicciones_con_intervalos(modelo, scaler, df)

def analizar_eventos_extremos(df):
    df['es_ola_calor'] = df['temperatura'] > df.groupby(['ciudad', 'mes'])['temperatura'].transform('mean') + 2 * df.groupby(['ciudad', 'mes'])['temperatura'].transform('std')
    eventos_por_año = df.groupby('año')['es_ola_calor'].sum().reset_index()
    eventos_por_año.columns = ['año', 'eventos_calor']
    return eventos_por_año

df_eventos = analizar_eventos_extremos(df)



# FUNCIONES AUXILIARES 

def calcular_nivel_alerta(cambio_temp):
    """Calcula el nivel de alerta según el cambio de temperatura"""
    if cambio_temp < 0.5:
        return "🟢 Normal", "success"
    elif cambio_temp < 1.0:
        return "🟡 Atención", "warning"
    elif cambio_temp < 1.5:
        return "🟠 Alerta", "warning"
    else:
        return "🔴 Crítico", "danger"

def generar_insight_automatico(df, temp_actual, cambio_temp):
    """Genera insights automáticos basados en los datos"""
    insights = []
    
    # Insight 1: Tendencia general
    if cambio_temp > 1.0:
        insights.append(f" **Alerta Climática**: La temperatura ha aumentado {cambio_temp:.2f}°C desde 1990, superando el objetivo del Acuerdo de París.")
    
    # Insight 2: Ciudad más afectada
    df_reciente = df[df['año'] >= 2020]
    ciudad_mas_caliente = df_reciente.groupby('ciudad')['temperatura'].mean().idxmax()
    temp_mas_alta = df_reciente.groupby('ciudad')['temperatura'].mean().max()
    insights.append(f" **Ciudad más cálida**: {ciudad_mas_caliente} con {temp_mas_alta:.1f}°C promedio (2020-2024)")
    
    # Insight 3: Aceleración
    cambio_decada_90 = df[df['año'].between(1990, 1999)]['temperatura'].mean()
    cambio_decada_20 = df[df['año'].between(2020, 2024)]['temperatura'].mean()
    aceleracion = cambio_decada_20 - cambio_decada_90
    insights.append(f" **Aceleración**: El calentamiento se ha intensificado en {aceleracion:.2f}°C entre los 90s y los 2020s")
    
    return insights


# SIDEBAR 

with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2E86AB 0%, #1E5F7E 100%); 
                padding: 1.5rem; border-radius: 8px; text-align: center; margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'> Cambio Climatico</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Dashboard Ejecutivo</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("###  Panel de Control")
    st.markdown("---")
    
    # Selector de año con preset
    preset_año = st.radio(
        "Período de análisis:",
        ["Actual (2024)", "Última década", "Histórico completo", "Personalizado"],
        index=0
    )
    
    if preset_año == "Personalizado":
        año_seleccionado = st.slider("Año específico:", 1990, 2050, 2024)
    elif preset_año == "Actual (2024)":
        año_seleccionado = 2024
    elif preset_año == "Última década":
        año_seleccionado = 2024
    else:
        año_seleccionado = 2024
    
    st.markdown("---")
    
    # Selector de ciudades 
    st.markdown("###  Selección de Ciudades")
    
    preset_ciudades = st.selectbox(
        "Presets rápidos:",
        ["Personalizado", "Top 5 Globales", "Hemisferio Norte", "Hemisferio Sur", "América", "Europa", "Asia"]
    )
    
    if preset_ciudades == "Top 5 Globales":
        ciudades_seleccionadas = ['Nueva York', 'Londres', 'Tokio', 'São Paulo', 'Mumbai']
    elif preset_ciudades == "Hemisferio Norte":
        ciudades_seleccionadas = ['Nueva York', 'Londres', 'Tokio', 'París', 'Moscú', 'Beijing', 'Toronto']
    elif preset_ciudades == "Hemisferio Sur":
        ciudades_seleccionadas = ['Sídney', 'São Paulo', 'Buenos Aires']
    elif preset_ciudades == "América":
        ciudades_seleccionadas = ['Nueva York', 'São Paulo', 'Ciudad de México', 'Buenos Aires', 'Toronto']
    elif preset_ciudades == "Europa":
        ciudades_seleccionadas = ['Londres', 'París', 'Moscú', 'Berlín']
    elif preset_ciudades == "Asia":
        ciudades_seleccionadas = ['Tokio', 'Mumbai', 'Beijing']
    else:
        ciudades_seleccionadas = st.multiselect(
            "Seleccionar ciudades:",
            options=list(CIUDADES_MUNDIALES.keys()),
            default=['Nueva York', 'Londres', 'Tokio', 'Buenos Aires']
        )
    
    st.markdown("---")
    
    # Opciones avanzadas
    with st.expander(" Opciones Avanzadas"):
        mostrar_intervalos = st.checkbox("Mostrar intervalos de confianza", value=True)
        mostrar_tendencias = st.checkbox("Destacar tendencias", value=True)
        modo_presentacion = st.checkbox("Modo presentación", value=False)
    
    st.markdown("---")
    
    # Información del modelo
    st.markdown("###  Modelo ML")
    st.metric("Precisión R²", f"{score*100:.1f}%")
    st.caption("Random Forest Regressor")
    
    st.markdown("---")
    
    # Exportar datos
    st.markdown("###  Exportar")
    df_export = df.groupby(['ciudad', 'año']).agg({
        'temperatura': ['mean', 'std', 'min', 'max'],
        'precipitacion': 'mean',
        'humedad': 'mean'
    }).round(2).reset_index()
    df_export.columns = ['ciudad', 'año', 'temp_promedio', 'temp_std', 'temp_min', 'temp_max', 'precip_promedio', 'humedad_promedio']
    csv = df_export.to_csv(index=False)
    st.download_button(
        label=" Descargar CSV",
        data=csv,
        file_name="clima_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    

# HEADER PRINCIPAL 

st.markdown("""
<div class="fade-in-up" style="text-align: center; margin-bottom: 2rem;">
    <h1 style="margin-bottom: 0.5rem;">  Dashboard Cambio Climatico </h1>
    <p style="font-size: 1.1rem; color: #718096; margin-bottom: 0;">
        Análisis Predictivo del Cambio Climático Global | 1990-2050
    </p>
    <p style="font-size: 0.9rem; color: #A0AEC0;">
        Powered by Machine Learning • Última actualización: """ + datetime.now().strftime("%d/%m/%Y %H:%M") + """
    </p>
</div>
""", unsafe_allow_html=True)


# DASHBOARD EJECUTIVO 

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Calcular métricas principales
temp_actual = df[df['año'] == 2024]['temperatura'].mean()
temp_1990 = df[df['año'] == 1990]['temperatura'].mean()
cambio_temp = temp_actual - temp_1990
temp_2050_pred = df_predicciones[df_predicciones['año'] == 2050]['temperatura'].mean()
eventos_2024 = df_eventos[df_eventos['año'] == 2024]['eventos_calor'].values[0] if len(df_eventos[df_eventos['año'] == 2024]) > 0 else 0
eventos_1990 = df_eventos[df_eventos['año'] == 1990]['eventos_calor'].values[0] if len(df_eventos[df_eventos['año'] == 1990]) > 0 else 0

nivel_alerta, tipo_alerta = calcular_nivel_alerta(cambio_temp)

# KPIs Ejecutivos
st.markdown("###  Indicadores Clave")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=" Temperatura Global 2024",
        value=f"{temp_actual:.2f}°C",
        delta=f"+{cambio_temp:.2f}°C vs 1990",
        delta_color="inverse"
    )

with col2:
    aumento_pct = (cambio_temp / temp_1990 * 100) if temp_1990 else 0
    st.metric(
        label=" Cambio Climático",
        value=f"+{cambio_temp:.2f}°C",
        delta=f"{aumento_pct:.1f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label=" Proyección 2050",
        value=f"{temp_2050_pred:.2f}°C",
        delta=f"+{(temp_2050_pred-temp_actual):.2f}°C",
        delta_color="inverse"
    )

with col4:
    cambio_eventos = ((eventos_2024 - eventos_1990) / eventos_1990 * 100) if eventos_1990 > 0 else 0
    st.metric(
        label=" Eventos Extremos",
        value=f"{eventos_2024}",
        delta=f"{cambio_eventos:+.0f}% vs 1990",
        delta_color="inverse"
    )

with col5:
    st.metric(
        label=" Precisión Modelo",
        value=f"{score*100:.1f}%",
        delta="R² Score"
    )

# Nivel de alerta
st.markdown(f"""
<div class="alert-box alert-{tipo_alerta}">
    <strong>{nivel_alerta}</strong> - 
    {'El aumento de temperatura está dentro de rangos esperados.' if cambio_temp < 1.0 else 
     'Atención: Se requiere monitoreo constante.' if cambio_temp < 1.5 else 
     'Situación crítica: Acciones inmediatas requeridas.'}
</div>
""", unsafe_allow_html=True)

# Insights automáticos
insights = generar_insight_automatico(df, temp_actual, cambio_temp)
st.markdown("""
<div class="insight-box">
    <div class="insight-title"> Insights Clave del Análisis</div>
""", unsafe_allow_html=True)
for insight in insights:
    st.markdown(f"• {insight}")
st.markdown("</div>", unsafe_allow_html=True)


# TABS PRINCIPALES

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Resumen Ejecutivo",
    " Análisis Histórico", 
    " Monitoreo en Tiempo Real",
    " Comparativas Regionales",
    " Proyecciones y Modelos",
])


# TAB 1 - RESUMEN EJECUTIVO 

with tab1:
    st.markdown("##  Resumen Ejecutivo")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico principal: Evolución histórica + predicción
        st.markdown("###  Trayectoria Climática Global")
        
        df_evol_hist = df.groupby('año')['temperatura'].mean().reset_index()
        df_evol_pred = df_predicciones.groupby('año')['temperatura'].mean().reset_index()
        
        fig_resumen = go.Figure()
        
        # Histórico
        fig_resumen.add_trace(go.Scatter(
            x=df_evol_hist['año'],
            y=df_evol_hist['temperatura'],
            name='Histórico',
            mode='lines',
            line=dict(color='#2E86AB', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
        
        # Predicción
        fig_resumen.add_trace(go.Scatter(
            x=df_evol_pred['año'],
            y=df_evol_pred['temperatura'],
            name='Predicción',
            mode='lines',
            line=dict(color='#F18F01', width=3, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(241, 143, 1, 0.1)'
        ))
        
        # Línea de referencia París (1.5°C)
        temp_base_1990 = df[df['año'] == 1990]['temperatura'].mean()
        fig_resumen.add_hline(
            y=temp_base_1990 + 1.5,
            line_dash="dot",
            line_color="red",
            annotation_text="Límite Acuerdo de París (+1.5°C)",
            annotation_position="right"
        )
        
        # Anotaciones de eventos clave
        fig_resumen.add_annotation(
            x=2015, y=df_evol_hist[df_evol_hist['año'] == 2015]['temperatura'].values[0],
            text="Acuerdo de París",
            showarrow=True,
            arrowhead=2,
            ax=-50, ay=-40,
            bgcolor="white",
            bordercolor="#2E86AB"
        )
        
        fig_resumen.update_layout(
            title='Evolución de Temperatura Global (1990-2050)',
            xaxis_title='Año',
            yaxis_title='Temperatura Promedio (°C)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_resumen, use_container_width=True)
        
        # Contexto científico
        st.markdown("""
        <div class="alert-box alert-info">
        <strong> Contexto Científico:</strong> El Acuerdo de París (2015) estableció el objetivo de limitar 
        el calentamiento global a 1.5°C por encima de los niveles preindustriales. Nuestros modelos muestran 
        que al ritmo actual, podríamos superar este umbral antes de 2040.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Comparación entre décadas
        st.markdown("###  Comparativa por Décadas")
        
        decadas_data = []
        for decada in [1990, 2000, 2010, 2020]:
            temp_decada = df[df['año'].between(decada, decada+9)]['temperatura'].mean()
            eventos_decada = df_eventos[df_eventos['año'].between(decada, decada+9)]['eventos_calor'].sum()
            decadas_data.append({
                'Década': f"{decada}s",
                'Temp °C': round(temp_decada, 2),
                'Eventos': int(eventos_decada)
            })
        
        df_decadas = pd.DataFrame(decadas_data)
        
        fig_decadas = go.Figure()
        fig_decadas.add_trace(go.Bar(
            x=df_decadas['Década'],
            y=df_decadas['Temp °C'],
            text=df_decadas['Temp °C'],
            textposition='outside',
            marker_color=['#A9CCE3', '#85C1E2', '#5DADE2', '#2E86AB']
        ))
        
        fig_decadas.update_layout(
            title='Temperatura Media por Década',
            yaxis_title='°C',
            template='plotly_white',
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_decadas, use_container_width=True)
        
        # Tabla resumen
        st.markdown("####  Datos Clave")
        st.dataframe(df_decadas, use_container_width=True, hide_index=True)
        
        # Ciudad más crítica
        st.markdown("####  Zona de Mayor Riesgo")
        df_2024 = df[df['año'] == 2024]
        ciudad_critica = df_2024.groupby('ciudad')['temperatura'].mean().idxmax()
        temp_critica = df_2024.groupby('ciudad')['temperatura'].mean().max()
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #C73E1D 0%, #922B21 100%); 
                    color: white; padding: 1rem; border-radius: 8px; text-align: center;">
            <h3 style="margin: 0; color: white;">{ciudad_critica}</h3>
            <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold;">{temp_critica:.1f}°C</p>
            <p style="margin: 0; font-size: 0.9rem;">Temperatura promedio más alta 2024</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    # Mapa de calor global
    st.markdown("###  Distribución Global de Temperaturas")
    
    df_mapa_resumen = df[df['año'] == 2024].groupby(['ciudad', 'lat', 'lon', 'pais']).agg({
        'temperatura': 'mean'
    }).reset_index()
    
    fig_mapa_resumen = go.Figure()
    
    fig_mapa_resumen.add_trace(go.Scattergeo(
        lon=df_mapa_resumen['lon'],
        lat=df_mapa_resumen['lat'],
        text=df_mapa_resumen['ciudad'],
        mode='markers',
        marker=dict(
            size=df_mapa_resumen['temperatura'] * 2,
            color=df_mapa_resumen['temperatura'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(
                title="°C",
                thickness=15,
                len=0.7
            ),
            line=dict(width=0.5, color='white')
        ),
        hovertemplate='<b>%{text}</b><br>Temperatura: %{marker.color:.1f}°C<extra></extra>'
    ))
    
    fig_mapa_resumen.update_layout(
        geo=dict(
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            projection_type='natural earth',
            showocean=True,
            oceancolor='rgb(230, 245, 255)',
            showcountries=True,
            countrycolor='rgb(204, 204, 204)'
        ),
        template='plotly_white',
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig_mapa_resumen, use_container_width=True)
    
    # Recomendaciones clave
    st.markdown("###  Recomendaciones Prioritarias")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid #06A77D; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #06A77D; margin-top: 0;"> Mitigación</h4>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>Reducción de emisiones 45% para 2030</li>
                <li>Transición energías renovables</li>
                <li>Reforestación acelerada</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid #F18F01; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #F18F01; margin-top: 0;"> Adaptación</h4>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>Infraestructura resiliente</li>
                <li>Sistemas de alerta temprana</li>
                <li>Planificación urbana sostenible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 8px; 
                    border-left: 4px solid #2E86AB; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #2E86AB; margin-top: 0;"> Monitoreo</h4>
            <ul style="margin: 0; padding-left: 1.2rem;">
                <li>Medición continua de KPIs</li>
                <li>Actualización de modelos ML</li>
                <li>Reportes trimestrales</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# TAB 2 - ANÁLISIS HISTÓRICO

with tab2:
    st.markdown("##  Análisis Histórico Detallado")
    st.markdown("---")
    
    # Heatmap mensual 
    st.markdown("###  Patrones de Calor Mensual")
    
    ciudad_heatmap = st.selectbox(
        "Seleccionar ciudad para análisis temporal:",
        list(CIUDADES_MUNDIALES.keys()),
        index=0
    )
    
    df_heatmap = df[df['ciudad'] == ciudad_heatmap].pivot_table(
        values='temperatura',
        index='mes',
        columns='año',
        aggfunc='mean'
    )
    
    # Nombres de meses
    meses_nombres = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 
                     'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=df_heatmap.values,
        x=df_heatmap.columns,
        y=meses_nombres,
        colorscale='RdYlBu_r',
        hovertemplate='Año: %{x}<br>Mes: %{y}<br>Temperatura: %{z:.1f}°C<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title=f'Temperatura Mensual Histórica - {ciudad_heatmap} (1990-2024)',
        xaxis_title='Año',
        yaxis_title='Mes',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por década
        st.markdown("###  Distribución de Temperaturas")
        
        df['decada'] = (df['año'] // 10) * 10
        
        fig_violin = go.Figure()
        
        for decada in sorted(df['decada'].unique()):
            df_decada = df[df['decada'] == decada]
            fig_violin.add_trace(go.Violin(
                y=df_decada['temperatura'],
                name=f'{int(decada)}s',
                box_visible=True,
                meanline_visible=True,
                fillcolor='lightblue',
                opacity=0.6,
                line_color='#2E86AB'
            ))
        
        fig_violin.update_layout(
            title='Distribución de Temperaturas por Década',
            yaxis_title='Temperatura (°C)',
            showlegend=True,
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with col2:
        # Análisis de variabilidad
        st.markdown("###  Variabilidad Climática")
        
        df_variabilidad = df.groupby('año').agg({
            'temperatura': ['mean', 'std']
        }).reset_index()
        df_variabilidad.columns = ['año', 'temp_media', 'temp_std']
        
        fig_variabilidad = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_variabilidad.add_trace(
            go.Scatter(
                x=df_variabilidad['año'],
                y=df_variabilidad['temp_media'],
                name='Temperatura Media',
                line=dict(color='#2E86AB', width=2)
            ),
            secondary_y=False
        )
        
        fig_variabilidad.add_trace(
            go.Scatter(
                x=df_variabilidad['año'],
                y=df_variabilidad['temp_std'],
                name='Desviación Estándar',
                line=dict(color='#F18F01', width=2, dash='dash')
            ),
            secondary_y=True
        )
        
        fig_variabilidad.update_layout(
            title='Media y Variabilidad de Temperatura',
            template='plotly_white',
            height=450,
            hovermode='x unified'
        )
        
        fig_variabilidad.update_xaxes(title_text='Año')
        fig_variabilidad.update_yaxes(title_text='Temperatura Media (°C)', secondary_y=False)
        fig_variabilidad.update_yaxes(title_text='Desviación Estándar', secondary_y=True)
        
        st.plotly_chart(fig_variabilidad, use_container_width=True)
    
    # Eventos extremos
    st.markdown("###  Análisis de Eventos Extremos")
    
    fig_eventos = go.Figure()
    
    fig_eventos.add_trace(go.Bar(
        x=df_eventos['año'],
        y=df_eventos['eventos_calor'],
        name='Eventos de Calor',
        marker_color='#C73E1D',
        hovertemplate='Año: %{x}<br>Eventos: %{y}<extra></extra>'
    ))
    
    # Tendencia
    z = np.polyfit(df_eventos['año'], df_eventos['eventos_calor'], 2)
    p = np.poly1d(z)
    fig_eventos.add_trace(go.Scatter(
        x=df_eventos['año'],
        y=p(df_eventos['año']),
        name='Tendencia',
        line=dict(color='#922B21', width=3, dash='dash')
    ))
    
    fig_eventos.update_layout(
        title='Frecuencia de Eventos de Calor Extremo (1990-2024)',
        xaxis_title='Año',
        yaxis_title='Número de Eventos',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig_eventos, use_container_width=True)
    
    # Estadísticas clave
    incremento_eventos = ((df_eventos[df_eventos['año'] >= 2020]['eventos_calor'].mean() - 
                          df_eventos[df_eventos['año'] < 2000]['eventos_calor'].mean()) / 
                         df_eventos[df_eventos['año'] < 2000]['eventos_calor'].mean() * 100)
    
    st.markdown(f"""
    <div class="alert-box alert-warning">
    <strong> Hallazgo Crítico:</strong> Los eventos de calor extremo han aumentado un 
    <strong>{incremento_eventos:.0f}%</strong> comparando los años 2020s con los 1990s. 
    Esta tendencia acelera el riesgo de sequías, incendios forestales y crisis de salud pública.
    </div>
    """, unsafe_allow_html=True)


# TAB 3 - TIEMPO REAL

with tab3:
    st.markdown("##  Monitoreo en Tiempo Real")
    st.markdown("---")
    
    col_refresh, col_time = st.columns([1, 3])
    with col_refresh:
        if st.button(" Actualizar Datos", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_time:
        st.info(f" Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    st.markdown("###  Condiciones Actuales por Ciudad")
    
    if ciudades_seleccionadas:
        # Grid de cards de ciudades
        cols = st.columns(min(3, len(ciudades_seleccionadas)))
        
        for idx, ciudad in enumerate(ciudades_seleccionadas):
            coords = CIUDADES_MUNDIALES[ciudad]
            datos_actuales = obtener_datos_actuales_api(ciudad, coords['lat'], coords['lon'])
            
            if datos_actuales:
                with cols[idx % 3]:
                    # Card personalizada
                    temp = datos_actuales['temperatura']
                    color_temp = '#C73E1D' if temp > 25 else '#F18F01' if temp > 15 else '#2E86AB'
                    
                    st.markdown(f"""
                    <div style="background: white; border-radius: 12px; padding: 1.5rem;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                                border-top: 4px solid {color_temp};">
                        <h3 style="margin-top: 0; color: #2D3748;">{ciudad}</h3>
                        <div style="font-size: 2.5rem; font-weight: bold; color: {color_temp}; margin: 1rem 0;">
                            {temp:.1f}°C
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; 
                                    font-size: 0.9rem; color: #718096;">
                            <div> Humedad: {datos_actuales['humedad']:.0f}%</div>
                            <div> Viento: {datos_actuales['viento']:.1f} km/h</div>
                            <div> Presión: {datos_actuales['presion']:.0f} hPa</div>
                            <div> {datos_actuales['descripcion'].title()}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Comparación con promedio histórico
        st.markdown("###  Comparación: Actual vs Histórico")
        
        df_2024 = df[df['año'] == 2024].groupby('ciudad')['temperatura'].mean().reset_index()
        
        datos_comparacion = []
        for ciudad in ciudades_seleccionadas:
            coords = CIUDADES_MUNDIALES[ciudad]
            datos_actuales = obtener_datos_actuales_api(ciudad, coords['lat'], coords['lon'])
            
            if datos_actuales:
                temp_actual = datos_actuales['temperatura']
                temp_hist = df_2024[df_2024['ciudad'] == ciudad]['temperatura'].values
                temp_hist = float(temp_hist[0]) if len(temp_hist) > 0 else temp_actual
                diferencia = temp_actual - temp_hist
                
                datos_comparacion.append({
                    'Ciudad': ciudad,
                    'Temp. Actual': f"{temp_actual:.1f}°C",
                    'Promedio 2024': f"{temp_hist:.1f}°C",
                    'Diferencia': f"{diferencia:+.1f}°C",
                    'Status': '🔴 Más cálido' if diferencia > 2 else '🟡 Normal' if abs(diferencia) <= 2 else '🔵 Más frío'
                })
        
        df_comparacion = pd.DataFrame(datos_comparacion)
        st.dataframe(df_comparacion, use_container_width=True, hide_index=True)
        
        # Gráfico de comparación
        fig_comparacion = go.Figure()
        
        temps_actuales = [float(d['Temp. Actual'].replace('°C', '')) for d in datos_comparacion]
        temps_hist = [float(d['Promedio 2024'].replace('°C', '')) for d in datos_comparacion]
        
        fig_comparacion.add_trace(go.Bar(
            name='Temperatura Actual',
            x=[d['Ciudad'] for d in datos_comparacion],
            y=temps_actuales,
            marker_color='#F18F01'
        ))
        
        fig_comparacion.add_trace(go.Bar(
            name='Promedio Histórico 2024',
            x=[d['Ciudad'] for d in datos_comparacion],
            y=temps_hist,
            marker_color='#2E86AB'
        ))
        
        fig_comparacion.update_layout(
            barmode='group',
            title='Comparación de Temperaturas',
            yaxis_title='Temperatura (°C)',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig_comparacion, use_container_width=True)
    
    else:
        st.warning(" Seleccioná al menos una ciudad en el panel lateral para ver datos en tiempo real.")


# TAB 4 - COMPARATIVAS

with tab4:
    st.markdown("##  Comparativas Regionales")
    st.markdown("---")
    
    if ciudades_seleccionadas:
        # Evolución comparativa
        st.markdown("###  Evolución Temporal Comparada")
        
        df_ciudades = df[df['ciudad'].isin(ciudades_seleccionadas)]
        df_ciudades_anual = df_ciudades.groupby(['año', 'ciudad'])['temperatura'].mean().reset_index()
        
        fig_ciudades = px.line(
            df_ciudades_anual,
            x='año',
            y='temperatura',
            color='ciudad',
            title='Evolución de Temperaturas por Ciudad (1990-2024)',
            labels={'temperatura': 'Temperatura (°C)', 'año': 'Año'},
            markers=True
        )
        
        fig_ciudades.update_traces(line=dict(width=3))
        fig_ciudades.update_layout(
            height=500,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_ciudades, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Comparación por hemisferio
            st.markdown("###  Análisis por Hemisferio")
            
            df['hemisferio'] = df['lat'].apply(lambda x: 'Hemisferio Norte' if x > 0 else 'Hemisferio Sur')
            df_hemisferio = df.groupby(['año', 'hemisferio'])['temperatura'].mean().reset_index()
            
            fig_hemisferio = px.area(
                df_hemisferio,
                x='año',
                y='temperatura',
                color='hemisferio',
                title='Comparación Norte vs Sur',
                labels={'temperatura': 'Temperatura (°C)', 'año': 'Año'},
                color_discrete_map={
                    'Hemisferio Norte': '#2E86AB',
                    'Hemisferio Sur': '#F18F01'
                }
            )
            
            fig_hemisferio.update_layout(
                height=450,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_hemisferio, use_container_width=True)

# TAB 5  RESUMEN ESTADISTICO Y CONCLUSIONES 

with tab5:
    st.header(" Resumen Estadístico y Conclusiones")

    col1, col2, col3 = st.columns(3)

    aumento_total = temp_2050_pred - temp_1990

    with col1:
        st.subheader(" Tendencias Históricas")
        st.write(f"*Aumento de temperatura (1990-2024):* {cambio_temp:.2f}°C")
        st.write(f"*Tasa promedio:* {(cambio_temp/34):.3f}°C por año")
        st.write(f"*Variabilidad:* {df[df['año'] >= 2020]['temperatura'].std():.2f}°C")

    with col2:
        st.subheader(" Proyecciones 2050")
        st.write(f"*Temperatura estimada 2050:* {temp_2050_pred:.2f}°C")
        st.write(f"*Aumento total (1990-2050):* {aumento_total:.2f}°C")
        st.write(f"*Precisión del modelo:* {score*100:.1f}%")

    with col3:
        st.subheader(" Datos del Análisis")
        st.write(f"*Ciudades analizadas:* {len(CIUDADES_MUNDIALES)}")
        st.write(f"*Período histórico:* 1990-2024")
        st.write(f"*Registros totales:* {len(df):,}")

    st.markdown("---")
    if st.sidebar.button(" Generar Reporte Ejecutivo"):
        with st.expander(" Reporte Ejecutivo - Cambio Climático", expanded=True):
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



