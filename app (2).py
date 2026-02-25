import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime
import warnings
import os  # Añadido para verificar existencia de archivos
warnings.filterwarnings('ignore')

# Configuración de la página - DEBE SER EL PRIMER COMANDO DE STREAMLIT
st.set_page_config(
    page_title="Analizador de Licitaciones - Residuos Peligrosos",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción principal
st.title("🗑️ Analizador de Licitaciones de Residuos Peligrosos")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>📊 Dashboard interactivo para el análisis de licitaciones públicas de retiro, traslado y disposición final de residuos peligrosos en Chile</h4>
    <p>Esta aplicación permite explorar en detalle las licitaciones adjudicadas entre 2019 y 2026, con filtros dinámicos y visualizaciones profesionales.</p>
    </div>
""", unsafe_allow_html=True)

# --- FUNCIONES DE PROCESAMIENTO ---

@st.cache_data
def cargar_y_procesar_datos(uploaded_file=None):
    """
    Carga y procesa los datos del archivo CSV
    - Si hay archivo subido, usa ese
    - Si no, busca el archivo por defecto en el repositorio
    """
    # Nombre del archivo por defecto
    archivo_por_defecto = 'ListaLicitaciones_filtrado_residuos_peligrosos_retiro_traslado.csv'
    
    if uploaded_file is not None:
        # Caso 1: Usuario subió un archivo
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        st.sidebar.success("✅ Archivo cargado manualmente")
        
    else:
        # Caso 2: Intentar cargar archivo por defecto del repositorio
        if os.path.exists(archivo_por_defecto):
            try:
                df = pd.read_csv(archivo_por_defecto, sep=';', encoding='utf-8')
                st.sidebar.success(f"✅ Archivo base cargado: {len(df)} licitaciones")
            except Exception as e:
                st.sidebar.error(f"❌ Error al cargar archivo por defecto: {e}")
                df = pd.DataFrame()
        else:
            st.sidebar.error(f"""
            ❌ No se encontró el archivo '{archivo_por_defecto}'
            
            Por favor, asegúrate de que el archivo existe en el repositorio.
            """)
            df = pd.DataFrame()
    
    # Si no hay datos, mostrar advertencia
    if df.empty:
        st.warning("⚠️ No hay datos para procesar. Por favor, sube un archivo CSV válido.")
        return df
    
    # Limpieza y procesamiento
    df['FechaPublicacion'] = pd.to_datetime(df['FechaPublicacion'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df['Año'] = df['FechaPublicacion'].dt.year
    df['Mes'] = df['FechaPublicacion'].dt.month
    df['MesNombre'] = df['FechaPublicacion'].dt.month_name().str[:3]
    df['Trimestre'] = df['FechaPublicacion'].dt.quarter
    df['Año-Mes'] = df['FechaPublicacion'].dt.to_period('M').astype(str)
    
    # Extraer región
    df['Region'] = df['Organismo'].apply(extraer_region)
    
    # Categorizar organismo
    df['CategoriaOrganismo'] = df['Organismo'].apply(categorizar_organismo)
    
    # Procesar montos
    df['Monto_Numérico_CLP'] = df['MontoLicitacion'].apply(extraer_monto_numerico)
    df['Monto_CLP_Millones'] = df['Monto_Numérico_CLP'] / 1_000_000
    df['Tipo_Monto_Categoria'] = df['MontoLicitacion'].apply(clasificar_tipo_monto)
    df['Monto_UTM_Estimado'] = df['MontoLicitacion'].apply(extraer_utm)
    
    return df

def extraer_region(organismo):
    """Extrae la región del nombre del organismo"""
    organismo_str = str(organismo).upper()
    
    regiones = {
        'Metropolitana': ['METROPOLITANA', 'SANTIAGO', 'MAIPU', 'SAN RAMON', 'RENCA', 'PROVIDENCIA', 
                         'LAS CONDES', 'NUNOA', 'LA CISTERNA', 'VITACURA', 'LO ESPEJO', 'CERRO NAVIA',
                         'CONCHALI', 'MACUL', 'LA REINA', 'PEÑAFLOR', 'EL MONTE', 'PAINE'],
        'Valparaíso': ['VALPARAISO', 'SAN FELIPE', 'VIÑA DEL MAR', 'QUILPUE', 'CARTAGENA', 'SAN ANTONIO',
                      'LOS ANDES', 'QUILLOTA', 'ZAPALLAR', 'NOGALES', 'LA LIGUA', 'PUCHUNCAVI',
                      'SAN ESTEBAN', 'CALLE LARGA', 'QUINTA DE TILCOCO', 'LIMACHE', 'SANTA MARIA'],
        'Biobío': ['BIO BIO', 'CONCEPCIÓN', 'TALCAHUANO', 'LOS ANGELES', 'CHIGUAYANTE', 'SAN PEDRO DE LA PAZ',
                  'CORONEL', 'LOTA', 'CURANILAHUE', 'MULCHEN', 'NACIMIENTO', 'YUNGAY', 'CABRERO'],
        'La Araucanía': ['ARAUCANIA', 'TEMUCO', 'ANGOL', 'VICTORIA', 'LAUTARO', 'NUEVA IMPERIAL',
                        'VILCUN', 'CUNCO', 'GORBEA', 'CURACAUTIN', 'LUMACO', 'CHOLCHOL'],
        'Los Lagos': ['LOS LAGOS', 'PUERTO MONTT', 'OSORNO', 'CASTRO', 'ANCUD', 'PUERTO VARAS',
                      'LLANQUIHUE', 'PALENA', 'CHILOE', 'CALBUCO', 'MAULLIN'],
        'Magallanes': ['MAGALLANES', 'PUNTA ARENAS', 'PORVENIR', 'PUERTO NATALES', 'PORVENIR'],
        'Coquimbo': ['COQUIMBO', 'LA SERENA', 'OVALLE', 'ILLAPEL', 'COMBARBALA', 'ANDACOLLO'],
        'Aysén': ['AYSEN', 'COYHAIQUE', 'PUERTO AYSEN', 'COCHRANE'],
        "O'Higgins": ['OHIGGINS', 'RANCAGUA', 'SAN FERNANDO', 'SAN VICENTE', 'PICHIDEGUA', 
                      'LAS CABRAS', 'PEUMO', 'COLTAUCO', 'DOÑIHUE', 'CODEGUA', 'MOSTAZAL', 'OLIVAR'],
        'Maule': ['MAULE', 'CURICO', 'TALCA', 'LINARES', 'CAUQUENES', 'CONSTITUCION', 
                  'PARRAL', 'SAN JAVIER', 'MOLINA', 'SAGRADA FAMILIA', 'PELARCO'],
        'Ñuble': ['ÑUBLE', 'CHILLAN', 'SAN CARLOS', 'BULNES', 'COBQUECURA', 'QUIRIHUE', 'COIHUECO'],
        'Arica y Parinacota': ['ARICA', 'PARINACOTA'],
        'Tarapacá': ['TARAPACA', 'IQUIQUE', 'ALTO HOSPICIO', 'POZO ALMONTE'],
        'Los Ríos': ['LOS RIOS', 'VALDIVIA', 'LA UNION', 'RIO BUENO', 'PANGUIPULLI'],
        'Atacama': ['ATACAMA', 'COPIAPO', 'VALLENAR', 'HUASCO', 'ALTO DEL CARMEN'],
        'Antofagasta': ['ANTOFAGASTA', 'CALAMA', 'TOCOPILLA', 'MARIA ELENA', 'OLLAGUE']
    }
    
    for region, keywords in regiones.items():
        if any(keyword in organismo_str for keyword in keywords):
            return region
    
    return 'Otra / Nacional'

def categorizar_organismo(nombre):
    """Categoriza el tipo de organismo"""
    nombre = str(nombre).upper()
    
    if any(word in nombre for word in ['MUNICIPALIDAD', 'ILUSTRE', 'I.', 'CORP MUNICIPAL', 'CORPORACION MUNICIPAL']):
        return 'Municipalidad / Corporación'
    elif any(word in nombre for word in ['SERVICIO DE SALUD', 'HOSPITAL', 'SUBSECRETARIA DE SALUD', 'SEREMI DE SALUD', 
                                         'CESFAM', 'CENTRO DE SALUD', 'CLINICA', 'COMPLEJO ASISTENCIAL']):
        return 'Salud'
    elif any(word in nombre for word in ['UNIVERSIDAD']):
        return 'Universidad'
    elif any(word in nombre for word in ['DIRECCION GENERAL DE AERONAUTICA CIVIL', 'DGAC']):
        return 'DGAC'
    elif any(word in nombre for word in ['FUERZA AEREA', 'EJERCITO', 'ARMADA', 'COMANDO']):
        return 'Fuerzas Armadas'
    elif any(word in nombre for word in ['MINISTERIO DE OBRAS PUBLICAS', 'VIALIDAD', 'MOP']):
        return 'MOP'
    else:
        return 'Otro Servicio Público'

def extraer_monto_numerico(monto_str):
    """Extrae un valor numérico del campo MontoLicitacion"""
    if pd.isna(monto_str) or monto_str in ['', 'nan']:
        return np.nan
    
    monto_str = str(monto_str).replace('.', '').replace(',', '').strip()
    
    # Si es un número puro
    if monto_str.isdigit():
        try:
            return float(monto_str)
        except:
            return np.nan
    
    # Si tiene UTM, extraer número y convertir (valor UTM aproximado)
    if 'UTM' in monto_str.upper():
        numeros = re.findall(r'[\d.]+', monto_str)
        if numeros:
            try:
                valor_utm = float(numeros[0].replace('.', ''))
                # Si hay un segundo número (ej: "Entre 100 y 1000 UTM"), usar el promedio
                if len(numeros) > 1:
                    valor_utm2 = float(numeros[1].replace('.', ''))
                    valor_utm = (valor_utm + valor_utm2) / 2
                return valor_utm * 60000  # Conversión aproximada
            except:
                return np.nan
    
    return np.nan

def extraer_utm(monto_str):
    """Extrae el valor en UTM si está presente"""
    monto_str = str(monto_str)
    if 'UTM' in monto_str.upper():
        numeros = re.findall(r'[\d.]+', monto_str)
        if numeros:
            try:
                valor_utm = float(numeros[0].replace('.', ''))
                # Si hay un segundo número, devolver el rango
                if len(numeros) > 1:
                    return f"{numeros[0]}-{numeros[1]} UTM"
                return f"{valor_utm} UTM"
            except:
                return np.nan
    return np.nan

def clasificar_tipo_monto(monto_str):
    """Clasifica el tipo de monto"""
    monto_str = str(monto_str)
    if 'UTM' in monto_str.upper():
        return 'Expresado en UTM'
    elif monto_str.replace('.', '').replace(',', '').isdigit():
        return 'Monto fijo en CLP'
    else:
        return 'Sin especificar'

# --- CARGA DE DATOS ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/waste--v1.png", width=100)
    st.header("⚙️ Configuración")
    
    uploaded_file = st.file_uploader(
        "Cargar archivo CSV de licitaciones",
        type=['csv'],
        help="Sube el archivo CSV con los datos de licitaciones. Si no subes ninguno, se usará el archivo base del repositorio."
    )
    
    # Cargar datos
    with st.spinner('Cargando y procesando datos...'):
        df = cargar_y_procesar_datos(uploaded_file)
    
    if not df.empty:
        st.success(f"✅ Datos cargados: {len(df)} licitaciones")
        
        # Mostrar info del dataset
        st.markdown("---")
        st.markdown("### 📊 Resumen del Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Años", f"{df['Año'].min()} - {df['Año'].max()}")
        with col2:
            st.metric("Regiones", df['Region'].nunique())
        
        st.markdown("---")
        st.header("🔍 Filtros")
        
        # Filtros interactivos
        años_disponibles = sorted(df['Año'].dropna().unique())
        años_seleccionados = st.multiselect(
            "Años",
            options=años_disponibles,
            default=años_disponibles if años_disponibles else []
        )
        
        regiones_disponibles = sorted(df['Region'].dropna().unique())
        regiones_seleccionadas = st.multiselect(
            "Regiones",
            options=regiones_disponibles,
            default=regiones_disponibles if regiones_disponibles else []
        )
        
        categorias_disponibles = sorted(df['CategoriaOrganismo'].dropna().unique())
        categorias_seleccionadas = st.multiselect(
            "Tipo de Organismo",
            options=categorias_disponibles,
            default=categorias_disponibles if categorias_disponibles else []
        )
        
        # Filtro de búsqueda por texto
        busqueda = st.text_input("🔎 Buscar en nombre u organismo", "")
        
        # Botón para aplicar filtros
        aplicar_filtros = st.button("🔄 Aplicar Filtros", type="primary")
    else:
        st.error("❌ No se pudieron cargar los datos")

# --- APLICAR FILTROS ---

if not df.empty:
    df_filtrado = df.copy()

    if años_seleccionados:
        df_filtrado = df_filtrado[df_filtrado['Año'].isin(años_seleccionados)]
    if regiones_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['Region'].isin(regiones_seleccionadas)]
    if categorias_seleccionadas:
        df_filtrado = df_filtrado[df_filtrado['CategoriaOrganismo'].isin(categorias_seleccionadas)]
    if busqueda:
        df_filtrado = df_filtrado[
            df_filtrado['NombreLicitacion'].str.contains(busqueda, case=False, na=False) |
            df_filtrado['Organismo'].str.contains(busqueda, case=False, na=False)
        ]

    # --- MÉTRICAS PRINCIPALES ---

    st.markdown("## 📈 Panel de Control")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_licitaciones = len(df_filtrado)
        st.metric(
            label="📋 Total Licitaciones",
            value=f"{total_licitaciones:,}",
            delta=f"{len(df_filtrado)/len(df)*100:.1f}% del total"
        )

    with col2:
        monto_total = df_filtrado['Monto_CLP_Millones'].sum()
        st.metric(
            label="💰 Monto Total (MM CLP)",
            value=f"${monto_total:,.0f}M" if not pd.isna(monto_total) else "N/A",
            delta="Suma de montos disponibles"
        )

    with col3:
        monto_promedio = df_filtrado['Monto_CLP_Millones'].mean()
        st.metric(
            label="📊 Monto Promedio (MM CLP)",
            value=f"${monto_promedio:,.1f}M" if not pd.isna(monto_promedio) else "N/A",
            delta="Por licitación"
        )

    with col4:
        organizaciones_unicas = df_filtrado['Organismo'].nunique()
        st.metric(
            label="🏢 Organizaciones",
            value=f"{organizaciones_unicas:,}",
            delta="Organismos distintos"
        )

    st.markdown("---")

    # --- VISUALIZACIONES PRINCIPALES ---

    # Crear pestañas para organizar el contenido
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Visión General",
        "🗺️ Análisis Regional",
        "🏛️ Análisis por Organismo",
        "📅 Tendencia Temporal",
        "📋 Datos Detallados"
    ])

    with tab1:
        st.header("Visión General del Mercado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución por región
            if not df_filtrado.empty and 'Region' in df_filtrado.columns:
                fig_regiones = px.pie(
                    df_filtrado,
                    names='Region',
                    title='Distribución de Licitaciones por Región',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_regiones.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_regiones, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar el gráfico")
        
        with col2:
            # Distribución por tipo de organismo
            if not df_filtrado.empty and 'CategoriaOrganismo' in df_filtrado.columns:
                cat_counts = df_filtrado['CategoriaOrganismo'].value_counts().reset_index()
                cat_counts.columns = ['CategoriaOrganismo', 'count']
                fig_categorias = px.bar(
                    cat_counts,
                    x='count',
                    y='CategoriaOrganismo',
                    title='Licitaciones por Tipo de Organismo',
                    orientation='h',
                    color='CategoriaOrganismo',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_categorias.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_categorias, use_container_width=True)
            else:
                st.info("No hay datos suficientes para mostrar el gráfico")
        
        # Evolución anual
        if not df_filtrado.empty and 'Año' in df_filtrado.columns:
            evolucion_anual = df_filtrado.groupby('Año').agg({
                'IDLicitacion': 'count',
                'Monto_CLP_Millones': 'sum'
            }).reset_index().rename(columns={'IDLicitacion': 'Cantidad'})
            
            fig_evolucion = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_evolucion.add_trace(
                go.Bar(x=evolucion_anual['Año'], y=evolucion_anual['Cantidad'], name="Cantidad", marker_color='#3498db'),
                secondary_y=False,
            )
            
            fig_evolucion.add_trace(
                go.Scatter(x=evolucion_anual['Año'], y=evolucion_anual['Monto_CLP_Millones'], 
                           name="Monto Total (MM CLP)", marker_color='#e74c3c', line=dict(width=3)),
                secondary_y=True,
            )
            
            fig_evolucion.update_layout(
                title_text="Evolución Anual de Licitaciones",
                hovermode='x unified'
            )
            fig_evolucion.update_xaxes(title_text="Año")
            fig_evolucion.update_yaxes(title_text="Cantidad de Licitaciones", secondary_y=False)
            fig_evolucion.update_yaxes(title_text="Monto Total (MM CLP)", secondary_y=True)
            
            st.plotly_chart(fig_evolucion, use_container_width=True)

    with tab2:
        st.header("Análisis Regional Detallado")
        
        if not df_filtrado.empty and 'Region' in df_filtrado.columns and len(df_filtrado['Region'].unique()) > 0:
            # Selector de región para análisis detallado
            region_analisis = st.selectbox(
                "Selecciona una región para análisis detallado",
                options=sorted(df_filtrado['Region'].unique())
            )
            
            df_region = df_filtrado[df_filtrado['Region'] == region_analisis]
            
            if not df_region.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Licitaciones en región", len(df_region))
                with col2:
                    monto_region = df_region['Monto_CLP_Millones'].sum()
                    st.metric("Monto total (MM CLP)", f"${monto_region:,.0f}M" if not pd.isna(monto_region) else "N/A")
                with col3:
                    st.metric("Organismos en región", df_region['Organismo'].nunique())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Top organismos en la región
                    top_organismos_region = df_region['Organismo'].value_counts().head(10)
                    if not top_organismos_region.empty:
                        fig_top_region = px.bar(
                            x=top_organismos_region.values,
                            y=top_organismos_region.index,
                            title=f'Top 10 Organismos en {region_analisis}',
                            orientation='h',
                            color=top_organismos_region.values,
                            color_continuous_scale='Viridis'
                        )
                        fig_top_region.update_layout(xaxis_title="Cantidad de Licitaciones", yaxis_title="")
                        st.plotly_chart(fig_top_region, use_container_width=True)
                
                with col2:
                    # Evolución en la región
                    evolucion_region = df_region.groupby('Año').size().reset_index(name='Cantidad')
                    if not evolucion_region.empty:
                        fig_evol_region = px.line(
                            evolucion_region,
                            x='Año',
                            y='Cantidad',
                            title=f'Evolución en {region_analisis}',
                            markers=True
                        )
                        fig_evol_region.update_layout(xaxis_title="Año", yaxis_title="Licitaciones")
                        st.plotly_chart(fig_evol_region, use_container_width=True)
                
                # Mapa de calor mensual
                heatmap_data = df_region.groupby(['Año', 'MesNombre']).size().reset_index(name='Cantidad')
                meses_orden = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                if not heatmap_data.empty:
                    fig_heatmap = px.density_heatmap(
                        heatmap_data,
                        x='Año',
                        y='MesNombre',
                        z='Cantidad',
                        title=f'Estacionalidad de Licitaciones en {region_analisis}',
                        color_continuous_scale='Reds',
                        category_orders={"MesNombre": meses_orden}
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No hay datos suficientes para el análisis regional")

    with tab3:
        st.header("Análisis por Organismo")
        
        if not df_filtrado.empty and 'CategoriaOrganismo' in df_filtrado.columns:
            # Selector de categoría
            categoria_analisis = st.selectbox(
                "Selecciona tipo de organismo",
                options=['Todos'] + sorted(df_filtrado['CategoriaOrganismo'].unique())
            )
            
            df_categoria = df_filtrado if categoria_analisis == 'Todos' else df_filtrado[df_filtrado['CategoriaOrganismo'] == categoria_analisis]
            
            if not df_categoria.empty:
                # Top organismos general
                st.subheader(f"Top 20 Organismos Licitantes - {categoria_analisis}")
                
                top_20 = df_categoria.groupby('Organismo').agg({
                    'IDLicitacion': 'count',
                    'Monto_CLP_Millones': 'sum'
                }).round(2).rename(columns={'IDLicitacion': 'Cantidad', 'Monto_CLP_Millones': 'Monto_Total_MM'})
                
                top_20 = top_20.sort_values('Cantidad', ascending=False).head(20).reset_index()
                
                if not top_20.empty:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_top = px.bar(
                            top_20,
                            x='Cantidad',
                            y='Organismo',
                            title='Por Cantidad de Licitaciones',
                            orientation='h',
                            color='Monto_Total_MM',
                            color_continuous_scale='Viridis',
                            text='Cantidad'
                        )
                        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
                        st.plotly_chart(fig_top, use_container_width=True)
                    
                    with col2:
                        # Tabla resumen
                        st.dataframe(
                            top_20[['Organismo', 'Cantidad', 'Monto_Total_MM']],
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Monto_Total_MM": st.column_config.NumberColumn(
                                    "Monto Total (MM CLP)",
                                    format="$ %.0fM"
                                )
                            }
                        )
                    
                    # Análisis de concentración
                    st.subheader("Análisis de Concentración del Mercado")
                    
                    # Calcular concentración (Top N %)
                    total_lic = len(df_categoria)
                    top_5_pct = (top_20.head(5)['Cantidad'].sum() / total_lic * 100) if total_lic > 0 else 0
                    top_10_pct = (top_20.head(10)['Cantidad'].sum() / total_lic * 100) if total_lic > 0 else 0
                    top_20_pct = (top_20['Cantidad'].sum() / total_lic * 100) if total_lic > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Concentración Top 5", f"{top_5_pct:.1f}%")
                    with col2:
                        st.metric("Concentración Top 10", f"{top_10_pct:.1f}%")
                    with col3:
                        st.metric("Concentración Top 20", f"{top_20_pct:.1f}%")
        else:
            st.info("No hay datos suficientes para el análisis por organismo")

    with tab4:
        st.header("Análisis de Tendencia Temporal")
        
        if not df_filtrado.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Vista por mes
                tendencia_mensual = df_filtrado.groupby(df_filtrado['FechaPublicacion'].dt.to_period('M')).size().reset_index(name='Cantidad')
                if not tendencia_mensual.empty:
                    tendencia_mensual['Fecha'] = tendencia_mensual['FechaPublicacion'].astype(str)
                    
                    fig_mensual = px.line(
                        tendencia_mensual,
                        x='Fecha',
                        y='Cantidad',
                        title='Tendencia Mensual de Licitaciones',
                        markers=True
                    )
                    fig_mensual.update_xaxes(title_text="Mes-Año")
                    fig_mensual.update_yaxes(title_text="Cantidad")
                    st.plotly_chart(fig_mensual, use_container_width=True)
            
            with col2:
                # Distribución por trimestre
                trimestres = df_filtrado.groupby(['Año', 'Trimestre']).size().reset_index(name='Cantidad')
                if not trimestres.empty:
                    trimestres['Año-Trim'] = trimestres['Año'].astype(str) + '-T' + trimestres['Trimestre'].astype(str)
                    
                    fig_trimestral = px.bar(
                        trimestres,
                        x='Año-Trim',
                        y='Cantidad',
                        title='Licitaciones por Trimestre',
                        color='Año',
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    fig_trimestral.update_xaxes(title_text="Año-Trimestre")
                    fig_trimestral.update_yaxes(title_text="Cantidad")
                    st.plotly_chart(fig_trimestral, use_container_width=True)
            
            # Análisis de estacionalidad
            st.subheader("Patrón Estacional por Mes")
            
            estacionalidad = df_filtrado.groupby('MesNombre').size().reset_index(name='Cantidad')
            meses_orden = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            if not estacionalidad.empty:
                fig_estacional = px.bar(
                    estacionalidad,
                    x='MesNombre',
                    y='Cantidad',
                    title='Distribución de Licitaciones por Mes',
                    color='Cantidad',
                    color_continuous_scale='Blues',
                    category_orders={"MesNombre": meses_orden}
                )
                fig_estacional.update_layout(xaxis_title="Mes", yaxis_title="Cantidad")
                st.plotly_chart(fig_estacional, use_container_width=True)
            
            # Análisis YoY (Year over Year)
            st.subheader("Crecimiento Interanual")
            
            yoy = df_filtrado.groupby('Año').size().reset_index(name='Cantidad')
            yoy['Crecimiento %'] = yoy['Cantidad'].pct_change() * 100
            
            fig_yoy = go.Figure()
            fig_yoy.add_trace(go.Bar(
                x=yoy['Año'],
                y=yoy['Cantidad'],
                name='Cantidad',
                marker_color='#2ecc71'
            ))
            fig_yoy.add_trace(go.Scatter(
                x=yoy['Año'],
                y=yoy['Crecimiento %'],
                name='Crecimiento %',
                yaxis='y2',
                marker_color='#e67e22',
                line=dict(width=3)
            ))
            
            fig_yoy.update_layout(
                title='Crecimiento Interanual de Licitaciones',
                xaxis=dict(title='Año'),
                yaxis=dict(title='Cantidad', side='left'),
                yaxis2=dict(title='Crecimiento %', side='right', overlaying='y', tickformat='.1f'),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_yoy, use_container_width=True)
        else:
            st.info("No hay datos suficientes para el análisis temporal")

    with tab5:
        st.header("Datos Detallados")
        
        if not df_filtrado.empty:
            # Selector de columnas a mostrar
            columnas_disponibles = ['IDLicitacion', 'NombreLicitacion', 'Tipo', 'Estado', 'FechaPublicacion',
                                   'Organismo', 'Region', 'CategoriaOrganismo', 'MontoLicitacion', 
                                   'Monto_CLP_Millones', 'Tipo_Monto_Categoria']
            
            columnas_mostrar = st.multiselect(
                "Selecciona columnas a mostrar",
                options=columnas_disponibles,
                default=['IDLicitacion', 'NombreLicitacion', 'Organismo', 'Region', 
                        'FechaPublicacion', 'MontoLicitacion']
            )
            
            if columnas_mostrar:
                df_display = df_filtrado[columnas_mostrar].copy()
                
                # Formatear fecha para mejor visualización
                if 'FechaPublicacion' in df_display.columns:
                    df_display['FechaPublicacion'] = df_display['FechaPublicacion'].dt.strftime('%d/%m/%Y')
                
                # Mostrar tabla con formato mejorado
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Monto_CLP_Millones": st.column_config.NumberColumn(
                            "Monto (MM CLP)",
                            format="$ %.2fM"
                        ),
                        "MontoLicitacion": st.column_config.TextColumn(
                            "Monto Original"
                        )
                    }
                )
                
                # Estadísticas y descargas
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Total registros:** {len(df_display)}")
                    if not df_filtrado['FechaPublicacion'].empty:
                        fecha_min = df_filtrado['FechaPublicacion'].min().strftime('%d/%m/%Y')
                        fecha_max = df_filtrado['FechaPublicacion'].max().strftime('%d/%m/%Y')
                        st.info(f"**Rango de fechas:** {fecha_min} a {fecha_max}")
                
                with col2:
                    # Botón de descarga
                    csv = df_display.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Descargar datos como CSV",
                        data=csv,
                        file_name=f"licitaciones_filtradas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
            else:
                st.warning("Selecciona al menos una columna para mostrar")
        else:
            st.info("No hay datos para mostrar")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 10px;'>
            <p>🗑️ Analizador de Licitaciones de Residuos Peligrosos | Desarrollado con Streamlit y Python</p>
            <p style='font-size: 0.8em;'>Datos actualizados al 24 de febrero de 2026</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.warning("👆 Por favor, sube un archivo CSV válido usando el botón en la barra lateral izquierda.")
