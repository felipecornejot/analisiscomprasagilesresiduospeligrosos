# 🗑️ Analizador de Licitaciones de Residuos Peligrosos

Dashboard interactivo desarrollado con **Streamlit** para el análisis de licitaciones públicas de retiro, traslado y disposición final de residuos peligrosos en Chile.

## 📋 Descripción

Esta aplicación permite explorar, filtrar y visualizar datos de licitaciones adjudicadas entre 2019 y 2026, proporcionando insights valiosos sobre el mercado de gestión de residuos peligrosos en el sector público chileno.

### 🔍 ¿Qué puedes hacer con esta herramienta?

- **Analizar tendencias** temporales de licitaciones
- **Comparar regiones** y su actividad en gestión de residuos
- **Identificar principales organismos** licitantes
- **Visualizar distribuciones** por tipo de organismo y región
- **Filtrar datos** de forma interactiva
- **Exportar datos** filtrados para análisis externo

## ✨ Características Principales

### 📊 Panel de Control
- Métricas clave: total de licitaciones, montos totales y promedio, organismos participantes
- Visualizaciones interactivas con Plotly
- Filtros dinámicos por año, región y tipo de organismo

### 🗺️ Análisis por Pestañas

| Pestaña | Descripción |
|---------|-------------|
| **Visión General** | Distribución geográfica, tipos de organismos y evolución anual |
| **Análisis Regional** | Desglose detallado por región con top organismos y estacionalidad |
| **Análisis por Organismo** | Ranking de licitantes y concentración del mercado |
| **Tendencia Temporal** | Patrones mensuales, trimestrales y crecimiento interanual |
| **Datos Detallados** | Tabla interactiva con exportación a CSV |

## 🛠️ Tecnologías Utilizadas

- **[Streamlit](https://streamlit.io/)** - Framework para aplicaciones de datos
- **[Pandas](https://pandas.pydata.org/)** - Manipulación y análisis de datos
- **[Plotly](https://plotly.com/python/)** - Visualizaciones interactivas
- **[NumPy](https://numpy.org/)** - Cálculos numéricos

## 📦 Instalación

### Requisitos previos
- Python 3.9 o superior
- pip (gestor de paquetes de Python)

### Pasos de instalación

1. **Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/licitaciones-respel.git
cd licitaciones-respel
