import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import random
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Configuración de la página
st.set_page_config(
    page_title = 'Proyecto IA - Consumo de Agua en CDMX',
    page_icon = '💧',
    layout = 'wide',
    initial_sidebar_state = 'expanded',
    menu_items = {
        'Get Help': None,
        'Report a Bug': None,
        'About': """
            **Integrantes del equipo:**\n
            * José Manuel Alonso Morales\n
            * Santiago Bañuelos Hernández\n
            * Emiliano Luna Casablanca\n
            [Repositorio de Github](https://github.com/JoseManuelAlonsoMorales/FinalProjectAI)
        """
    }
)

# Cargamos los datos desde un archivo CSV
class Application:
    def __init__(self, path):
        self.__df = pd.read_csv(path)
        self.__colonias = self.__df['colonia'].dropna().unique().tolist()
        self.__alcaldias = self.__df['alcaldia'].dropna().unique().tolist()
        self.__consumo_total = self.__df['consumo_total'].unique()
        
    def getDataFrame(self):
        return self.__df

    def getColonias(self):
        return self.__colonias
    
    def getAlcaldias(self):
        return self.__alcaldias
    
    def getConsumoTotal(self):
        return self.__consumo_total
    
    def getListaColonias(self):
        return self.data['colonia'].tolist()
    
    def getListaAlcaldias(self):
        return self.data['alcaldia'].tolist()
    
    def getListaConsumoTotal(self):
        return self.data['consumo_total'].tolist()

    def getConsumoDataFrame(self):
        return self.__df_consumo

    def getReportes2024DataFrame(self):
        return self.__df_reportes_2024

    def getReportesHistoricoDataFrame(self):
        return self.__df_reportes_hist

    def getColonias(self):
        return self.__colonias

    def getAlcaldias(self):
        return self.__alcaldias

    def getConsumoTotal(self):
        return self.__consumo_total

    def getListaColonias(self):
        return self.data['colonia'].tolist()

    def getListaAlcaldias(self):
        return self.data['alcaldia'].tolist()

    def getListaConsumoTotal(self):
        return self.data['consumo_total'].tolist()
    
    def limpiarDataFrame(self):
        # Limpiamos valores nulos y cadenas vacías en alcaldía y colonia
        self.data = self.__df.dropna(subset=['alcaldia', 'colonia'])

        # Convertimos alcaldía y colonia a tipo string y eliminamos los espacios en blanco
        self.data['alcaldia'] = self.data['alcaldia'].astype(str).str.strip()
        self.data['colonia'] = self.data['colonia'].astype(str).str.strip()

        # Eliminamos filas en donde el valor de la alcaldía sea nan o una cadena vacía
        self.data = self.data[(self.data['alcaldia'] != '') & (self.data['alcaldia'].notna())]
    
    # Creamos una lista para almacenar la cantidad de agua transportada por colonia
    def generarAguaTransportada(self, seed=2004, max_value=15000):
        consumo = self.data["consumo_total"]
        random.seed(seed)
        return [random.randint(int(consumo.min()), max_value) for _ in range(len(consumo))]

    # Diccionario para almacenar las alcaldías y sus colonias con los datos de transporte y consumo
    def getDiccionarioAlcaldiasColonias(self):
        colonias = self.getListaColonias()
        alcaldias = self.getListaAlcaldias()
        consumo_total = self.data["consumo_total"].tolist()
        agua_transportada = self.generarAguaTransportada()

        dicc = {}
        for i in range(len(alcaldias)):
            alcaldia = alcaldias[i]
            colonia = colonias[i]
            transporte = agua_transportada[i]
            consumo = consumo_total[i]

            if alcaldia not in dicc:
                dicc[alcaldia] = {}
            if colonia in dicc[alcaldia]:
                dicc[alcaldia][colonia][0].append(transporte)
                dicc[alcaldia][colonia][1].append(consumo)
            else:
                dicc[alcaldia][colonia] = [[transporte], [consumo]]

        return dicc

def filtroAlcaldiaColonia(app):
    st.sidebar.title("Filtros")

    alcaldias = sorted(list(app.getAlcaldias()))
    alcaldias = ["-- Todas las alcaldías --"] + alcaldias
    alcaldia_seleccionada = st.sidebar.selectbox("Selecciona una alcaldía", alcaldias)

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        colonias_filtradas = app.data[app.data["alcaldia"] == alcaldia_seleccionada]["colonia"].unique()
        colonias = sorted(colonias_filtradas)
        colonias = ["-- Todas las colonias --"] + colonias
        colonia_seleccionada = st.sidebar.selectbox("Selecciona una colonia:", colonias)
    else:
        colonia_seleccionada = "-- Todas las colonias --"
    
    return alcaldia_seleccionada, colonia_seleccionada

def mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada):
    # Subheader de los filtros aplicados
    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        if colonia_seleccionada == "-- Todas las colonias --":
            st.subheader(f"Datos en {alcaldia_seleccionada}")
        else:
            st.subheader(f"Datos en {alcaldia_seleccionada} para la colonia {colonia_seleccionada}")

# Inicializamos la aplicación y cargamos los datos
app = Application(r'https://raw.githubusercontent.com/JoseManuelAlonsoMorales/FinalProjectAI/main/data/consumo_agua_historico_2019.csv')  # URL del archivo CSV
df = app.getDataFrame()

# Cargamos reportes de agua 2024 y reportes históricos
df_reportes_2024 = pd.read_csv('data/reportes_agua_2024_01.csv')
df_reportes_hist = pd.read_csv('data/reportes_agua_hist.csv')

# Normaliza columnas para facilitar merge
df_reportes_2024 = df_reportes_2024.rename(columns={'alcaldia_catalogo':'alcaldia','colonia_catalogo':'colonia'})
df_reportes_hist = df_reportes_hist.rename(columns={'alcaldia':'alcaldia','colonia_datos_abiertos':'colonia'})

df_reportes = pd.concat([
    df_reportes_2024[['latitud','longitud','alcaldia','colonia']],
    df_reportes_hist[['latitud','longitud','alcaldia','colonia']]
], ignore_index=True)

reportes_group = df_reportes.groupby(['latitud','longitud']).size().reset_index(name='num_reportes')

st.title('Análisis del Consumo de Agua en CDMX')

# Comenzamos a trabajar con los datos
app.limpiarDataFrame()  # Limpiamos el DataFrame de valores nulos y cadenas vacías

cant_consumida_max_min = np.array(app.getListaConsumoTotal()) # Convertimos la lista de consumo total a un array de numpy para obtener los valores máximos y mínimos
diccionario_alcaldias_colonias = app.getDiccionarioAlcaldiasColonias() # Obtenemos el diccionario de alcaldías y colonias con los datos de transporte y consumo

# Opciones de análisis
tabs = st.tabs(["Introducción", "Datos", "Análisis de Regresión", "Clasificación y Segmentación", "Mapa de Consumo"])

alcaldia_seleccionada, colonia_seleccionada = filtroAlcaldiaColonia(app)

# Entrenamiento general del modelo de clasificación (con todos los datos del diccionario)
ArrayDatos = []
DatosY = []

for alcaldia in diccionario_alcaldias_colonias:
    for colonia in diccionario_alcaldias_colonias[alcaldia]:
        transporte_list = diccionario_alcaldias_colonias[alcaldia][colonia][0]
        consumo_list = diccionario_alcaldias_colonias[alcaldia][colonia][1]

        for transporte, consumo in zip(transporte_list, consumo_list):
            ArrayDatos.append([transporte, consumo])
            diferencia = transporte - consumo
            promedio = (transporte + consumo) / 2

            if diferencia <= 0:
                DatosY.append(0)  # Peligro
            elif diferencia >= promedio:
                DatosY.append(2)  # Perfecto
            else:
                DatosY.append(1)  # Medio

Xclasificacion = np.array(ArrayDatos)
YClasificacion = np.array(DatosY)

modelo_general_knn = KNeighborsClassifier(n_neighbors=20)
modelo_general_knn.fit(Xclasificacion, YClasificacion)

# Si el usuario selecciona "Introducción", mostramos un resumen de los datos
with tabs[0]:
    st.markdown("""
        ## Análisis Integral del Consumo de Agua en la CDMX

        El agua es un recurso vital y estratégico para el desarrollo sostenible de la Ciudad de México, una metrópoli con más de 9 millones de habitantes que enfrenta retos complejos relacionados con su distribución, acceso y consumo.

        Este proyecto se basa en un exhaustivo análisis de datos geoespaciales y temporales para comprender cómo se distribuye y consume el agua en las distintas colonias de la ciudad. Se incluyen variables detalladas sobre el consumo de agua segmentado por tipo de usuario —doméstico, no doméstico y mixto—, así como indicadores sociodemográficos y económicos que permiten contextualizar el uso del recurso en cada zona.

        A través de visualizaciones interactivas y herramientas analíticas, buscamos:

        - Detectar zonas con fallas o limitaciones en el suministro de agua.
        - Identificar patrones y disparidades en los niveles de consumo entre regiones.
        - Evaluar el impacto de factores como el Índice de Desarrollo Social, la densidad poblacional y la infraestructura urbana.
        - Ofrecer una base sólida para la toma de decisiones en políticas públicas y gestión hídrica.

        Los datos abarcan múltiples periodos, lo que facilita el seguimiento de tendencias temporales y la evaluación de intervenciones o cambios en las políticas de agua. Este análisis contribuye a promover un uso más eficiente y equitativo del agua, apuntando hacia la sustentabilidad y mejora en la calidad de vida de los habitantes de la Ciudad de México.

        En suma, este proyecto es una herramienta clave para técnicos, autoridades y ciudadanos interesados en la gestión responsable de uno de los recursos más importantes para nuestra ciudad y nuestro futuro.
        """)



# Si el usuario selecciona "Datos", mostramos el DataFrame
with tabs[1]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)

    # Aplicar filtros según selección
    df_filtrado = app.data

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        df_filtrado = df_filtrado[df_filtrado["alcaldia"] == alcaldia_seleccionada]

    if colonia_seleccionada != "-- Todas las colonias --":
        df_filtrado = df_filtrado[df_filtrado["colonia"] == colonia_seleccionada]
    
    st.dataframe(df_filtrado)

    st.markdown("""
        En esta sección puedes explorar el conjunto de datos que contiene información detallada sobre el consumo de agua en distintas colonias y alcaldías de la Ciudad de México.

        El DataFrame incluye las siguientes columnas principales:

        - **fecha_referencia:** Fecha del registro del consumo.
        - **anio:** Año al que corresponde el dato.
        - **bimestre:** Periodo bimestral del año.
        - **consumo_total_mixto:** Consumo total de agua para usuarios mixtos (litros).
        - **consumo_prom_dom:** Consumo promedio de agua para usuarios domésticos (litros).
        - **consumo_total_dom:** Consumo total de agua para usuarios domésticos (litros).
        - **consumo_prom_mixto:** Consumo promedio para usuarios mixtos (litros).
        - **consumo_total:** Consumo total combinado (litros).
        - **consumo_prom:** Consumo promedio combinado (litros).
        - **consumo_prom_no_dom:** Consumo promedio para usuarios no domésticos (litros).
        - **consumo_total_no_dom:** Consumo total para usuarios no domésticos (litros).
        - **indice_des:** Índice de desarrollo social asociado a la colonia.
        - **colonia:** Nombre de la colonia.
        - **alcaldia:** Nombre de la alcaldía.
        - **latitud:** Coordenada geográfica de latitud.
        - **longitud:** Coordenada geográfica de longitud.

        Esta información permite un análisis detallado y localizado del consumo hídrico, facilitando la identificación de patrones y posibles áreas con retos en el suministro de agua.
        """)


# Si el usuario selecciona "Análisis de Regresión", mostramos el modelo de regresión lineal
with tabs[2]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)

    # Recolectar datos según filtros
    datos_transporte = []
    datos_consumo = []

    if alcaldia_seleccionada == "-- Todas las alcaldías --":
        # Agregamos todos los datos de todas las alcaldías y colonias
        for alcaldia in diccionario_alcaldias_colonias:
            for colonia in diccionario_alcaldias_colonias[alcaldia]:
                datos_transporte.extend(diccionario_alcaldias_colonias[alcaldia][colonia][0])
                datos_consumo.extend(diccionario_alcaldias_colonias[alcaldia][colonia][1])
        
    elif colonia_seleccionada == "-- Todas las colonias --":
        for colonia in diccionario_alcaldias_colonias[alcaldia_seleccionada]:
            datos_transporte.extend(diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia][0])
            datos_consumo.extend(diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia][1])
        
    else:
        datos_transporte = diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia_seleccionada][0]
        datos_consumo = diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia_seleccionada][1]

    # Validar datos
    if len(datos_transporte) > 1 and len(datos_transporte) == len(datos_consumo):
        X = np.array(datos_transporte).reshape(-1, 1)
        Y = np.array(datos_consumo)

        modelo = LinearRegression()
        modelo.fit(X, Y)
        y_pred = modelo.predict(X)

        df_plot = pd.DataFrame({
            "Agua Transportada": datos_transporte,
            "Consumo de Agua": datos_consumo,
            "Predicción de Consumo": y_pred
        })

        fig = px.scatter(df_plot, x="Agua Transportada", y="Consumo de Agua", 
                         color_discrete_sequence=["blue"])
        fig.add_scatter(mode="lines", name="Línea de Regresión", line=dict(color="red"),
                        x=df_plot["Agua Transportada"], y=df_plot["Predicción de Consumo"])
        
        st.plotly_chart(fig)

    else:
        st.warning("No hay suficientes datos para entrenar el modelo.")

    st.markdown("""
        ### Análisis con Regresión Lineal

        La regresión lineal es un modelo estadístico que busca encontrar la relación entre dos variables cuantitativas. En este caso, analizamos cómo el **agua transportada** influye en el **consumo de agua**.

        - **Los puntos en el gráfico** representan observaciones individuales: cada punto muestra un par de valores de agua transportada y consumo de agua correspondientes a una colonia o alcaldía.
        - **La línea roja (pendiente)** es la línea de regresión, que indica la tendencia general de los datos. Esta línea minimiza la distancia entre ella y todos los puntos.
        - **La pendiente de la línea** nos muestra cómo cambia el consumo esperado cuando cambia la cantidad de agua transportada:
        - Si la pendiente es positiva (hacia arriba), significa que a mayor agua transportada, mayor es el consumo de agua esperado.
        - Si la pendiente es negativa (hacia abajo), indicaría que un aumento en el agua transportada se asocia con una disminución en el consumo, lo cual sería poco común en este contexto.
        
        Este análisis nos ayuda a entender y predecir el comportamiento del consumo en función del suministro de agua, facilitando la toma de decisiones para una mejor gestión del recurso.
        """)

# Si el usuario selecciona "Clasificación y Segmentación", mostramos el modelo de clasificación
with tabs[3]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)
    
    # Construcción de datos de entrada
    registros = []

    for alcaldia, colonias_dict in diccionario_alcaldias_colonias.items():
        if alcaldia_seleccionada != "-- Todas las alcaldías --" and alcaldia != alcaldia_seleccionada:
            continue

        for colonia, valores in colonias_dict.items():
            if colonia_seleccionada != "-- Todas las colonias --" and colonia != colonia_seleccionada:
                continue

            transporte_list = valores[0]
            consumo_list = valores[1]

            for transporte, consumo in zip(transporte_list, consumo_list):
                diferencia = transporte - consumo
                promedio = (transporte + consumo) / 2

                if diferencia <= 0:
                    categoria = "Peligro"
                elif diferencia >= promedio:
                    categoria = "Perfecto"
                else:
                    categoria = "Medio"

                registros.append({
                    "Agua Transportada": transporte,
                    "Consumo de Agua": consumo,
                    "Categoría": categoria
                })

    # Verificación
    if not registros:
        st.warning("No hay datos disponibles para la selección.")
    else:
        df_clasificacion = pd.DataFrame(registros)

        # Clasificación usando el modelo previamente entrenado
        X = df_clasificacion[["Agua Transportada", "Consumo de Agua"]].values
        y_pred = modelo_general_knn.predict(X)

        reverse_map = {0: "Peligro", 1: "Medio", 2: "Perfecto"}
        df_clasificacion["Categoría"] = [reverse_map[i] for i in y_pred]

        # Visualización con plotly express
        fig = px.scatter(
            df_clasificacion,
            x="Agua Transportada",
            y="Consumo de Agua",
            color="Categoría",
            symbol="Categoría",
            labels={
                "Agua Transportada": "Agua Transportada (litros)",
                "Consumo de Agua": "Consumo de Agua (litros)"
            }
        )

        st.plotly_chart(fig)

    st.markdown("""
        ### Análisis con Modelo de Clasificación

        En esta sección utilizamos un modelo de clasificación basado en el algoritmo **K-Nearest Neighbors (KNN)**, que clasifica las observaciones en categorías según sus características.

        - El modelo toma como entrada dos variables: **agua transportada** y **consumo de agua**.
        - Clasificamos cada punto en tres categorías:
        - **Perfecto**: cuando el agua transportada es significativamente mayor al consumo, indicando un suministro adecuado o exceso.
        - **Medio**: cuando el agua transportada es similar al consumo, mostrando un balance adecuado pero con poca holgura.
        - **Peligro**: cuando el agua transportada es menor o igual al consumo, lo que puede reflejar un riesgo de insuficiencia o desabasto.

        El algoritmo KNN asigna la categoría a cada punto basándose en la similitud con sus vecinos más cercanos, lo que permite identificar zonas con diferentes niveles de suministro y consumo, facilitando la toma de decisiones para mejorar la distribución del agua.
        """)

with tabs[4]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)

    opciones_mapa = st.multiselect(
        "Selecciona qué mostrar en el mapa:",
        ['Consumo de Agua', 'Número de Reportes'],
        default=['Consumo de Agua']
    )

    df_consumo_filtrado = app.data.copy()
    df_reportes_filtrado = reportes_group.copy()

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        df_consumo_filtrado = df_consumo_filtrado[df_consumo_filtrado['alcaldia'] == alcaldia_seleccionada]
        df_reportes_filtrado = df_reportes_filtrado.merge(
            df_reportes[df_reportes['alcaldia'] == alcaldia_seleccionada][['latitud','longitud']],
            on=['latitud','longitud'], how='inner'
        )
    if colonia_seleccionada != "-- Todas las colonias --":
        df_consumo_filtrado = df_consumo_filtrado[df_consumo_filtrado['colonia'] == colonia_seleccionada]
        df_reportes_filtrado = df_reportes_filtrado.merge(
            df_reportes[df_reportes['colonia'] == colonia_seleccionada][['latitud','longitud']],
            on=['latitud','longitud'], how='inner'
        )

    if df_consumo_filtrado.empty and df_reportes_filtrado.empty:
        st.warning("No hay datos disponibles para la selección.")
    else:
        m = folium.Map(location=[19.4326, -99.1332], zoom_start=10)

        if 'Consumo de Agua' in opciones_mapa and not df_consumo_filtrado.empty:
            heat_data = [
                [row['latitud'], row['longitud'], row['consumo_total']]
                for idx, row in df_consumo_filtrado.iterrows()
                if not pd.isnull(row['latitud']) and not pd.isnull(row['longitud'])
            ]
            HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)

        if 'Número de Reportes' in opciones_mapa and not df_reportes_filtrado.empty:
            for idx, row in df_reportes_filtrado.iterrows():
                lat = row['latitud']
                lon = row['longitud']
                num = row['num_reportes']

                if pd.isnull(lat) or pd.isnull(lon):
                    continue

                radius = 5 + min(num, 30)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6,
                    popup=f"Reportes: {num}",
                    tooltip=f"Reportes: {num}"
                ).add_to(m)

        st_folium(m, width=700, height=500)

        st.markdown("""
            ### Mapa Interactivo: Consumo y Reportes de Agua en CDMX

            - El mapa de calor muestra el consumo total de agua en las distintas áreas.
            - Las burbujas rojas indican la cantidad de reportes en esa ubicación, con tamaño proporcional a la cantidad.
            - Usa el selector para visualizar uno o ambos indicadores y explorar las zonas críticas.
        """)
