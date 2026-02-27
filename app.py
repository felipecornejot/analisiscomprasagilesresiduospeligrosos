import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import os
import traceback
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Financiero - Resumen Ingresos/Egresos",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("💰 Dashboard Financiero - Resumen de Ingresos y Egresos")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4>📊 Análisis detallado de la evolución financiera de empresas por mes</h4>
    <p>Visualización interactiva de saldos, ingresos y egresos para cada entidad.</p>
    </div>
""", unsafe_allow_html=True)

# --- FUNCIONES DE PROCESAMIENTO ---

@st.cache_data
def cargar_datos_financieros(uploaded_file=None):
    """
    Carga y procesa los datos del archivo Excel con múltiples hojas
    """
    archivo_por_defecto = 'Resumen ingresos-egresos.xlsx'
    
    if uploaded_file is not None:
        df_dict = pd.read_excel(uploaded_file, sheet_name=None, header=None)
        st.sidebar.success("✅ Archivo cargado manualmente")
    else:
        if os.path.exists(archivo_por_defecto):
            try:
                df_dict = pd.read_excel(archivo_por_defecto, sheet_name=None, header=None)
                st.sidebar.success(f"✅ Archivo base cargado: {len(df_dict)} meses")
            except Exception as e:
                st.sidebar.error(f"❌ Error al cargar archivo: {e}")
                return None
        else:
            st.sidebar.error(f"❌ No se encontró el archivo '{archivo_por_defecto}'")
            return None
    
    # Procesar cada hoja (mes)
    datos_completos = []
    
    for mes, df in df_dict.items():
        try:
            # Limpiar y procesar el DataFrame
            df_limpio = df.iloc[4:8, 1:13].copy()  # Filas 5-8, columnas B-M (índices 1-12)
            df_limpio.columns = ['Recauchaje Insamar', 'Banco Chile_1', 'Logística', 'Sustrend', 
                                'Sustrend Laboratorios', 'Volltech', 'Dario E.I.R.L.', 
                                'Sangha Inmobiliaria', 'Banco Chile_2', 'Inversiones Sangha', 
                                'Wellnes Academy', 'Banco Santander Stgo']
            
            # Asignar índices (filas)
            df_limpio.index = ['Saldo inicial', 'Ingresos', 'Egresos', 'Saldo final']
            
            # Transformar a formato largo
            df_melted = df_limpio.T.reset_index()
            df_melted.columns = ['Empresa', 'Saldo_inicial', 'Ingresos', 'Egresos', 'Saldo_final']
            df_melted['Mes'] = mes
            
            # Limpiar valores (convertir a numérico)
            for col in ['Saldo_inicial', 'Ingresos', 'Egresos', 'Saldo_final']:
                df_melted[col] = pd.to_numeric(df_melted[col], errors='coerce')
            
            # Eliminar filas con todos NaN o empresas sin datos
            df_melted = df_melted.dropna(subset=['Saldo_inicial', 'Ingresos', 'Egresos', 'Saldo_final'], how='all')
            df_melted = df_melted[~df_melted['Empresa'].str.contains('Banco', na=False)]
            
            datos_completos.append(df_melted)
            
        except Exception as e:
            st.warning(f"Error procesando mes {mes}: {e}")
            continue
    
    if datos_completos:
        df_final = pd.concat(datos_completos, ignore_index=True)
        
        # Ordenar meses cronológicamente
        orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                      'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        df_final['Mes'] = pd.Categorical(df_final['Mes'], categories=orden_meses, ordered=True)
        df_final = df_final.sort_values(['Mes', 'Empresa'])
        
        # Calcular métricas adicionales
        df_final['Resultado_neto'] = df_final['Ingresos'] + df_final['Egresos']  # Egresos son negativos
        df_final['Variacion_saldo'] = df_final['Saldo_final'] - df_final['Saldo_inicial']
        df_final['Margen'] = (df_final['Resultado_neto'] / df_final['Ingresos'].replace(0, np.nan)) * 100
        
        # Limpiar valores infinitos y NaN
        df_final = df_final.replace([np.inf, -np.inf], np.nan)
        
        return df_final
    else:
        return None

def formatear_moneda(valor):
    """Formatea valores como moneda chilena"""
    if pd.isna(valor):
        return "N/A"
    return f"${valor:,.0f}"

def safe_plotly_chart(fig, key):
    """Función segura para mostrar gráficos de Plotly"""
    try:
        st.plotly_chart(fig, use_container_width=True, key=key)
    except Exception as e:
        st.error(f"Error al mostrar el gráfico: {str(e)}")
        st.info("Mostrando datos en formato tabla como alternativa")
        return False
    return True

# --- CARGA DE DATOS ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/money--v1.png", width=100)
    st.header("⚙️ Configuración")
    
    uploaded_file = st.file_uploader(
        "Cargar archivo Excel",
        type=['xlsx'],
        help="Sube el archivo 'Resumen ingresos-egresos.xlsx'",
        key="file_uploader"
    )
    
    with st.spinner('Cargando y procesando datos...'):
        df = cargar_datos_financieros(uploaded_file)
    
    if df is not None and not df.empty:
        st.success(f"✅ Datos cargados: {df['Mes'].nunique()} meses, {df['Empresa'].nunique()} empresas")
        
        # Mostrar info del dataset
        st.markdown("---")
        st.markdown("### 📊 Resumen")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Meses", df['Mes'].nunique())
        with col2:
            st.metric("Empresas", df['Empresa'].nunique())
        
        st.markdown("---")
        st.header("🔍 Filtros")
        
        # Filtros
        meses_disponibles = sorted(df['Mes'].unique())
        meses_seleccionados = st.multiselect(
            "📅 Meses",
            options=meses_disponibles,
            default=meses_disponibles,
            key="filtro_meses"
        )
        
        empresas_disponibles = sorted(df['Empresa'].unique())
        empresas_seleccionadas = st.multiselect(
            "🏢 Empresas",
            options=empresas_disponibles,
            default=empresas_disponibles,
            key="filtro_empresas"
        )
        
        # Rango de valores
        st.markdown("### 💰 Filtros de montos")
        
        # Manejar valores NaN para los sliders
        saldo_max = df['Saldo_final'].max()
        if pd.isna(saldo_max):
            saldo_max = 0
            
        ingreso_max = df['Ingresos'].max()
        if pd.isna(ingreso_max):
            ingreso_max = 0
        
        col1, col2 = st.columns(2)
        with col1:
            rango_saldo = st.slider(
                "Saldo final (millones)",
                min_value=0.0,
                max_value=float(saldo_max/1_000_000) if saldo_max > 0 else 100.0,
                value=(0.0, float(saldo_max/1_000_000) if saldo_max > 0 else 100.0),
                step=1.0,
                key="rango_saldo"
            )
        with col2:
            rango_ingreso = st.slider(
                "Ingresos (millones)",
                min_value=0.0,
                max_value=float(ingreso_max/1_000_000) if ingreso_max > 0 else 100.0,
                value=(0.0, float(ingreso_max/1_000_000) if ingreso_max > 0 else 100.0),
                step=1.0,
                key="rango_ingreso"
            )
    else:
        st.error("❌ No se pudieron cargar los datos")
        st.stop()

# --- APLICAR FILTROS ---

df_filtrado = df.copy()

if meses_seleccionados:
    df_filtrado = df_filtrado[df_filtrado['Mes'].isin(meses_seleccionados)]
if empresas_seleccionadas:
    df_filtrado = df_filtrado[df_filtrado['Empresa'].isin(empresas_seleccionadas)]
if rango_saldo:
    df_filtrado = df_filtrado[
        (df_filtrado['Saldo_final'].fillna(0).abs() >= rango_saldo[0]*1_000_000) &
        (df_filtrado['Saldo_final'].fillna(0).abs() <= rango_saldo[1]*1_000_000)
    ]
if rango_ingreso:
    df_filtrado = df_filtrado[
        (df_filtrado['Ingresos'].fillna(0).abs() >= rango_ingreso[0]*1_000_000) &
        (df_filtrado['Ingresos'].fillna(0).abs() <= rango_ingreso[1]*1_000_000)
    ]

# Verificar que hay datos después de filtrar
if df_filtrado.empty:
    st.warning("⚠️ No hay datos que cumplan con los filtros seleccionados.")
    st.stop()

# --- MÉTRICAS PRINCIPALES ---

st.markdown("## 📈 Panel de Control Financiero")

col1, col2, col3, col4 = st.columns(4)

with col1:
    saldo_total = df_filtrado.groupby('Mes')['Saldo_final'].sum().sum()
    st.metric(
        label="💰 Saldo Total Acumulado",
        value=formatear_moneda(saldo_total),
        delta=f"Promedio: {formatear_moneda(saldo_total/df_filtrado['Mes'].nunique())}"
    )

with col2:
    ingresos_totales = df_filtrado.groupby('Mes')['Ingresos'].sum().sum()
    st.metric(
        label="📈 Ingresos Totales",
        value=formatear_moneda(ingresos_totales),
        delta=f"Promedio mes: {formatear_moneda(ingresos_totales/df_filtrado['Mes'].nunique())}"
    )

with col3:
    egresos_totales = df_filtrado.groupby('Mes')['Egresos'].sum().sum()
    st.metric(
        label="📉 Egresos Totales",
        value=formatear_moneda(abs(egresos_totales)),
        delta=f"Promedio mes: {formatear_moneda(abs(egresos_totales)/df_filtrado['Mes'].nunique())}",
        delta_color="inverse"
    )

with col4:
    resultado_neto = ingresos_totales + egresos_totales
    st.metric(
        label="💵 Resultado Neto",
        value=formatear_moneda(resultado_neto),
        delta=f"Margen: {(resultado_neto/ingresos_totales*100):.1f}%" if ingresos_totales != 0 else "N/A"
    )

st.markdown("---")

# --- VISUALIZACIONES PRINCIPALES ---

# Crear pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visión General",
    "🏢 Análisis por Empresa",
    "📅 Evolución Temporal",
    "💰 Comparativa",
    "📋 Datos Detallados"
])

with tab1:
    st.header("Visión General del Portfolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Saldo por empresa (último mes disponible)
        ultimo_mes = df_filtrado['Mes'].max()
        df_ultimo = df_filtrado[df_filtrado['Mes'] == ultimo_mes].copy()
        
        if not df_ultimo.empty and df_ultimo['Saldo_final'].notna().any():
            fig_saldo = px.bar(
                df_ultimo.dropna(subset=['Saldo_final']),
                x='Empresa',
                y='Saldo_final',
                title=f'Saldo Final por Empresa - {ultimo_mes}',
                color='Saldo_final',
                color_continuous_scale='RdYlGn',
                text_auto='.2s'
            )
            fig_saldo.update_layout(xaxis_tickangle=-45)
            safe_plotly_chart(fig_saldo, "bar_saldo")
        else:
            st.info("No hay datos para el último mes")
    
    with col2:
        # Distribución de ingresos vs egresos
        df_melt = df_filtrado.melt(
            id_vars=['Empresa', 'Mes'],
            value_vars=['Ingresos', 'Egresos'],
            var_name='Tipo',
            value_name='Monto'
        )
        df_melt['Monto_abs'] = df_melt['Monto'].abs()
        df_melt = df_melt.dropna(subset=['Monto_abs'])
        
        if not df_melt.empty:
            fig_dist = px.box(
                df_melt,
                x='Tipo',
                y='Monto_abs',
                color='Tipo',
                title='Distribución de Ingresos y Egresos',
                points='all',
                color_discrete_map={'Ingresos': '#2ecc71', 'Egresos': '#e74c3c'}
            )
            fig_dist.update_layout(yaxis_title="Monto ($)")
            safe_plotly_chart(fig_dist, "box_dist")
        else:
            st.info("No hay datos suficientes")
    
    # Top empresas por ingresos
    st.subheader("Top 5 Empresas por Ingresos")
    top_ingresos = df_filtrado.groupby('Empresa')['Ingresos'].sum().nlargest(5).reset_index()
    top_ingresos = top_ingresos.dropna()
    
    if not top_ingresos.empty and top_ingresos['Ingresos'].sum() > 0:
        fig_top = px.bar(
            top_ingresos,
            x='Ingresos',
            y='Empresa',
            title='Empresas con mayores ingresos totales',
            orientation='h',
            color='Ingresos',
            color_continuous_scale='Viridis',
            text_auto='.2s'
        )
        safe_plotly_chart(fig_top, "bar_top")
    else:
        st.info("No hay datos suficientes")

with tab2:
    st.header("Análisis Detallado por Empresa")
    
    # Selector de empresa
    empresa_seleccionada = st.selectbox(
        "Selecciona una empresa",
        options=df_filtrado['Empresa'].unique(),
        key="select_empresa"
    )
    
    df_empresa = df_filtrado[df_filtrado['Empresa'] == empresa_seleccionada].copy()
    df_empresa = df_empresa.dropna(subset=['Saldo_final', 'Ingresos', 'Egresos'])
    
    if not df_empresa.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ultimo_saldo = df_empresa['Saldo_final'].iloc[-1] if not df_empresa['Saldo_final'].empty else 0
            st.metric("Saldo actual", formatear_moneda(ultimo_saldo))
        with col2:
            st.metric("Ingresos totales", formatear_moneda(df_empresa['Ingresos'].sum()))
        with col3:
            st.metric("Egresos totales", formatear_moneda(abs(df_empresa['Egresos'].sum())))
        with col4:
            st.metric("Resultado neto", formatear_moneda(df_empresa['Resultado_neto'].sum()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Evolución mensual
            if not df_empresa[['Ingresos', 'Egresos', 'Saldo_final']].isna().all().all():
                fig_evol = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_evol.add_trace(
                    go.Bar(x=df_empresa['Mes'], y=df_empresa['Ingresos'], name="Ingresos", marker_color='#2ecc71'),
                    secondary_y=False
                )
                fig_evol.add_trace(
                    go.Bar(x=df_empresa['Mes'], y=df_empresa['Egresos'], name="Egresos", marker_color='#e74c3c'),
                    secondary_y=False
                )
                fig_evol.add_trace(
                    go.Scatter(x=df_empresa['Mes'], y=df_empresa['Saldo_final'], 
                              name="Saldo final", line=dict(color='#3498db', width=3)),
                    secondary_y=True
                )
                
                fig_evol.update_layout(
                    title=f'Evolución mensual - {empresa_seleccionada}',
                    hovermode='x unified'
                )
                fig_evol.update_xaxes(title_text="Mes")
                fig_evol.update_yaxes(title_text="Ingresos/Egresos ($)", secondary_y=False)
                fig_evol.update_yaxes(title_text="Saldo final ($)", secondary_y=True)
                
                safe_plotly_chart(fig_evol, f"line_evol_empresa_{empresa_seleccionada}")
            else:
                st.info("No hay datos de evolución mensual")
        
        with col2:
            # Composición
            total_ingresos = df_empresa['Ingresos'].sum()
            total_egresos = abs(df_empresa['Egresos'].sum())
            
            if (total_ingresos > 0 or total_egresos > 0) and not (pd.isna(total_ingresos) or pd.isna(total_egresos)):
                fig_pie = px.pie(
                    values=[total_ingresos, total_egresos],
                    names=['Ingresos', 'Egresos'],
                    title=f'Composición - {empresa_seleccionada}',
                    color_discrete_map={'Ingresos': '#2ecc71', 'Egresos': '#e74c3c'},
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                safe_plotly_chart(fig_pie, f"pie_empresa_{empresa_seleccionada}")
            else:
                st.info("Sin datos de ingresos/egresos")
        
        # Tabla de datos mensuales
        st.subheader("Datos mensuales")
        df_display = df_empresa[['Mes', 'Saldo_inicial', 'Ingresos', 'Egresos', 'Saldo_final', 'Resultado_neto']].copy()
        for col in ['Saldo_inicial', 'Ingresos', 'Egresos', 'Saldo_final', 'Resultado_neto']:
            df_display[col] = df_display[col].apply(formatear_moneda)
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info(f"No hay datos disponibles para {empresa_seleccionada}")

with tab3:
    st.header("Evolución Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Evolución del saldo total
        saldo_mensual = df_filtrado.groupby('Mes')['Saldo_final'].sum().reset_index()
        saldo_mensual = saldo_mensual.dropna()
        
        if not saldo_mensual.empty and saldo_mensual['Saldo_final'].notna().any():
            fig_saldo_mensual = px.line(
                saldo_mensual,
                x='Mes',
                y='Saldo_final',
                title='Evolución del Saldo Total',
                markers=True,
                line_shape='spline'
            )
            fig_saldo_mensual.update_traces(line=dict(color='#3498db', width=3))
            fig_saldo_mensual.update_layout(yaxis_title="Saldo total ($)")
            safe_plotly_chart(fig_saldo_mensual, "line_saldo_mensual")
        else:
            st.info("No hay datos suficientes")
    
    with col2:
        # Ingresos vs Egresos por mes
        flujo_mensual = df_filtrado.groupby('Mes').agg({
            'Ingresos': 'sum',
            'Egresos': 'sum'
        }).reset_index()
        flujo_mensual = flujo_mensual.dropna()
        
        if not flujo_mensual.empty and (flujo_mensual['Ingresos'].notna().any() or flujo_mensual['Egresos'].notna().any()):
            fig_flujo = go.Figure()
            fig_flujo.add_trace(go.Bar(x=flujo_mensual['Mes'], y=flujo_mensual['Ingresos'], 
                                       name='Ingresos', marker_color='#2ecc71'))
            fig_flujo.add_trace(go.Bar(x=flujo_mensual['Mes'], y=flujo_mensual['Egresos'], 
                                       name='Egresos', marker_color='#e74c3c'))
            
            fig_flujo.update_layout(
                title='Ingresos vs Egresos por Mes',
                barmode='group',
                yaxis_title="Monto ($)"
            )
            safe_plotly_chart(fig_flujo, "bar_flujo")
        else:
            st.info("No hay datos suficientes")
    
    # Heatmap de saldos por empresa y mes
    st.subheader("Mapa de Calor - Saldos por Empresa y Mes")
    
    pivot_saldos = df_filtrado.pivot_table(
        values='Saldo_final',
        index='Empresa',
        columns='Mes',
        aggfunc='first'
    ).fillna(0)
    
    if not pivot_saldos.empty and pivot_saldos.values.max() > 0:
        fig_heatmap = px.imshow(
            pivot_saldos,
            title='Saldos (pesos)',
            color_continuous_scale='RdYlGn',
            aspect='auto',
            text_auto='.0f'
        )
        fig_heatmap.update_layout(xaxis_title="Mes", yaxis_title="Empresa")
        safe_plotly_chart(fig_heatmap, "heatmap_saldos")
    else:
        st.info("No hay datos suficientes para el mapa de calor")

with tab4:
    st.header("Análisis Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ranking de rentabilidad
        rentabilidad = df_filtrado.groupby('Empresa').agg({
            'Resultado_neto': 'sum',
            'Ingresos': 'sum'
        }).reset_index()
        rentabilidad = rentabilidad.replace([np.inf, -np.inf], np.nan).dropna()
        rentabilidad['Margen'] = (rentabilidad['Resultado_neto'] / rentabilidad['Ingresos'].replace(0, np.nan) * 100).round(1)
        rentabilidad = rentabilidad.sort_values('Resultado_neto', ascending=False)
        rentabilidad = rentabilidad[rentabilidad['Resultado_neto'].notna()]
        
        if not rentabilidad.empty and rentabilidad['Resultado_neto'].abs().sum() > 0:
            fig_rent = px.bar(
                rentabilidad.head(10),
                x='Resultado_neto',
                y='Empresa',
                title='Ranking de Rentabilidad (Resultado Neto)',
                orientation='h',
                color='Margen',
                color_continuous_scale='RdYlGn',
                text_auto='.2s'
            )
            fig_rent.update_layout(xaxis_title="Resultado neto ($)")
            safe_plotly_chart(fig_rent, "bar_rentabilidad")
        else:
            st.info("No hay datos de rentabilidad")
    
    with col2:
        # Participación por empresa
        participacion = df_filtrado.groupby('Empresa')['Ingresos'].sum().sort_values(ascending=False)
        participacion = participacion[participacion > 0].dropna()
        
        if not participacion.empty and participacion.sum() > 0:
            fig_part = px.pie(
                values=participacion.values,
                names=participacion.index,
                title='Participación en Ingresos Totales',
                hole=0.4
            )
            fig_part.update_traces(textposition='inside', textinfo='percent+label')
            safe_plotly_chart(fig_part, "pie_participacion")
        else:
            st.info("No hay datos de participación")
    
    # Gráfico de comparación - VERSIÓN SIMPLIFICADA Y SEGURA
    st.subheader("Comparación Ingresos vs Egresos por Empresa")
    
    try:
        # Preparar datos de forma segura
        resumen_empresas = df_filtrado.groupby('Empresa').agg({
            'Ingresos': 'sum',
            'Egresos': 'sum'
        }).reset_index()
        
        # Limpiar datos
        resumen_empresas = resumen_empresas.dropna()
        resumen_empresas['Egresos_abs'] = resumen_empresas['Egresos'].abs()
        resumen_empresas = resumen_empresas[resumen_empresas['Ingresos'] > 0]
        
        if not resumen_empresas.empty and len(resumen_empresas) >= 1:
            # Usar gráfico de barras agrupadas (MUCHO MÁS ESTABLE)
            fig_comparativa = go.Figure()
            
            fig_comparativa.add_trace(go.Bar(
                x=resumen_empresas['Empresa'],
                y=resumen_empresas['Ingresos'],
                name='Ingresos',
                marker_color='#2ecc71'
            ))
            
            fig_comparativa.add_trace(go.Bar(
                x=resumen_empresas['Empresa'],
                y=resumen_empresas['Egresos_abs'],
                name='Egresos',
                marker_color='#e74c3c'
            ))
            
            fig_comparativa.update_layout(
                title='Ingresos vs Egresos por Empresa',
                xaxis_title='Empresa',
                yaxis_title='Monto ($)',
                barmode='group',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_comparativa, use_container_width=True, key="bar_comparativa_estable")
            
            # Mostrar tabla de datos
            with st.expander("Ver datos detallados"):
                st.dataframe(resumen_empresas[['Empresa', 'Ingresos', 'Egresos', 'Egresos_abs']])
        else:
            st.info("No hay suficientes datos para la comparativa")
            
            # Mostrar raw data como fallback
            st.dataframe(df_filtrado[['Empresa', 'Ingresos', 'Egresos']].head(10))
    
    except Exception as e:
        st.warning(f"Error al generar el gráfico: {str(e)}")
        # Mostrar datos en tabla como último recurso
        st.dataframe(df_filtrado.groupby('Empresa')[['Ingresos', 'Egresos']].sum().reset_index())

with tab5:
    st.header("Datos Detallados")
    
    if not df_filtrado.empty:
        # Selector de columnas
        columnas_disponibles = ['Mes', 'Empresa', 'Saldo_inicial', 'Ingresos', 'Egresos', 
                                'Saldo_final', 'Resultado_neto', 'Variacion_saldo', 'Margen']
        
        columnas_mostrar = st.multiselect(
            "Selecciona columnas a mostrar",
            options=columnas_disponibles,
            default=['Mes', 'Empresa', 'Ingresos', 'Egresos', 'Saldo_final', 'Resultado_neto'],
            key="select_columnas"
        )
        
        if columnas_mostrar:
            df_display = df_filtrado[columnas_mostrar].copy()
            
            # Formatear números
            for col in ['Saldo_inicial', 'Ingresos', 'Egresos', 'Saldo_final', 'Resultado_neto', 'Variacion_saldo']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(formatear_moneda)
            
            if 'Margen' in df_display.columns:
                df_display['Margen'] = df_display['Margen'].round(1).astype(str) + '%'
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                key="dataframe_detallado"
            )
            
            # Estadísticas y descarga
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Total registros:** {len(df_display)}")
                st.info(f"**Meses:** {df_filtrado['Mes'].nunique()}")
                st.info(f"**Empresas:** {df_filtrado['Empresa'].nunique()}")
            
            with col2:
                # Botón de descarga
                csv = df_filtrado.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Descargar datos completos como CSV",
                    data=csv,
                    file_name=f"datos_financieros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary",
                    key="download_button"
                )
    else:
        st.info("No hay datos para mostrar")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 10px;'>
        <p>💰 Dashboard Financiero - Resumen de Ingresos y Egresos | Desarrollado con Streamlit y Python</p>
        <p style='font-size: 0.8em;'>Datos actualizados al {}</p>
    </div>
""".format(datetime.now().strftime('%d/%m/%Y')), unsafe_allow_html=True)
