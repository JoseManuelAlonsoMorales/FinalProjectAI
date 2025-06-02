import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import random
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

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

# Cargamos los datos
class Application:
    def __init__(self, consumo_path, reportes_2024_path, reportes_hist_path):
        try:
            self.__df_consumo = pd.read_csv(consumo_path)
        except FileNotFoundError:
            st.error(f"Error: No se encontró el archivo de consumo en {consumo_path}")
            self.__df_consumo = pd.DataFrame() # Empty DataFrame
        try:
            self.__df_reportes_2024 = pd.read_csv(reportes_2024_path)
        except FileNotFoundError:
            st.error(f"Error: No se encontró el archivo de reportes 2024 en {reportes_2024_path}")
            self.__df_reportes_2024 = pd.DataFrame()
        try:
            self.__df_reportes_hist = pd.read_csv(reportes_hist_path)
        except FileNotFoundError:
            st.error(f"Error: No se encontró el archivo de reportes históricos en {reportes_hist_path}")
            self.__df_reportes_hist = pd.DataFrame()

        # Atributos para los dataframes limpios
        self.data = pd.DataFrame()
        self.reportes_2024_data = pd.DataFrame()
        self.reportes_hist_data = pd.DataFrame()

        if not self.__df_consumo.empty:
            self.__colonias = self.__df_consumo['colonia'].dropna().unique().tolist()
            self.__alcaldias = self.__df_consumo['alcaldia'].dropna().unique().tolist()
            self.__consumo_total_unique = self.__df_consumo['consumo_total'].unique() # Renamed to avoid conflict
        else:
            self.__colonias = []
            self.__alcaldias = []
            self.__consumo_total_unique = []


    def getConsumoDataFrame(self): # Original df
        return self.__df_consumo

    def getReportes2024DataFrame(self): # Original df
        return self.__df_reportes_2024

    def getReportesHistoricoDataFrame(self): # Original df
        return self.__df_reportes_hist

    def getColonias(self): # From consumption data
        return self.__colonias

    def getAlcaldias(self): # From consumption data
        return self.__alcaldias

    def getConsumoTotalUnique(self): # From consumption data
        return self.__consumo_total_unique

    def getListaColonias(self): # From cleaned consumption data (self.data)
        if not self.data.empty:
            return self.data['colonia'].tolist()
        return []

    def getListaAlcaldias(self): # From cleaned consumption data (self.data)
        if not self.data.empty:
            return self.data['alcaldia'].tolist()
        return []

    def getListaConsumoTotal(self): # From cleaned consumption data (self.data)
        if not self.data.empty:
            return self.data['consumo_total'].tolist()
        return []

    def limpiar_dataframes(self):
        # Limpiar DataFrame de Consumo
        if not self.__df_consumo.empty:
            self.data = self.__df_consumo.copy()
            self.data.dropna(subset=['alcaldia', 'colonia', 'latitud', 'longitud'], inplace=True)
            self.data['alcaldia'] = self.data['alcaldia'].astype(str).str.strip()
            self.data['colonia'] = self.data['colonia'].astype(str).str.strip()
            self.data = self.data[(self.data['alcaldia'] != '') & (self.data['alcaldia'].notna())]
            self.data = self.data[(self.data['colonia'] != '') & (self.data['colonia'].notna())]
             # Asegurar que latitud y longitud sean numéricos
            self.data['latitud'] = pd.to_numeric(self.data['latitud'], errors='coerce')
            self.data['longitud'] = pd.to_numeric(self.data['longitud'], errors='coerce')
            self.data.dropna(subset=['latitud', 'longitud'], inplace=True)


        # Limpiar DataFrame de Reportes 2024
        if not self.__df_reportes_2024.empty:
            self.reportes_2024_data = self.__df_reportes_2024.copy()
            self.reportes_2024_data.dropna(subset=['latitud', 'longitud', 'alcaldia_catalogo', 'colonia_catalogo'], inplace=True)
            self.reportes_2024_data.rename(columns={'alcaldia_catalogo': 'alcaldia', 'colonia_catalogo': 'colonia'}, inplace=True)
            self.reportes_2024_data['alcaldia'] = self.reportes_2024_data['alcaldia'].astype(str).str.strip()
            self.reportes_2024_data['colonia'] = self.reportes_2024_data['colonia'].astype(str).str.strip()
            self.reportes_2024_data['latitud'] = pd.to_numeric(self.reportes_2024_data['latitud'], errors='coerce')
            self.reportes_2024_data['longitud'] = pd.to_numeric(self.reportes_2024_data['longitud'], errors='coerce')
            self.reportes_2024_data.dropna(subset=['latitud', 'longitud'], inplace=True)

        # Limpiar DataFrame de Reportes Históricos
        if not self.__df_reportes_hist.empty:
            self.reportes_hist_data = self.__df_reportes_hist.copy()
            self.reportes_hist_data.dropna(subset=['latitud', 'longitud', 'alcaldia'], inplace=True)
            
            # Unificar columnas de colonia
            if 'colonia_datos_abiertos' in self.reportes_hist_data.columns and 'colonia_registro_sacmex' in self.reportes_hist_data.columns:
                self.reportes_hist_data['colonia'] = self.reportes_hist_data['colonia_datos_abiertos'].fillna(self.reportes_hist_data['colonia_registro_sacmex'])
            elif 'colonia_datos_abiertos' in self.reportes_hist_data.columns:
                self.reportes_hist_data['colonia'] = self.reportes_hist_data['colonia_datos_abiertos']
            elif 'colonia_registro_sacmex' in self.reportes_hist_data.columns:
                self.reportes_hist_data['colonia'] = self.reportes_hist_data['colonia_registro_sacmex']
            else:
                self.reportes_hist_data['colonia'] = np.nan # O manejar de otra forma si ninguna existe
            
            self.reportes_hist_data.dropna(subset=['colonia'], inplace=True)
            self.reportes_hist_data['alcaldia'] = self.reportes_hist_data['alcaldia'].astype(str).str.strip()
            self.reportes_hist_data['colonia'] = self.reportes_hist_data['colonia'].astype(str).str.strip()
            self.reportes_hist_data['latitud'] = pd.to_numeric(self.reportes_hist_data['latitud'], errors='coerce')
            self.reportes_hist_data['longitud'] = pd.to_numeric(self.reportes_hist_data['longitud'], errors='coerce')
            self.reportes_hist_data.dropna(subset=['latitud', 'longitud'], inplace=True)

    def generarAguaTransportada(self, seed=2004, max_value=15000):
        if self.data.empty or 'consumo_total' not in self.data.columns or self.data['consumo_total'].empty:
            return []
        consumo = self.data["consumo_total"]
        if consumo.min() > max_value : # Evitar error en randint si min > max_value
             min_val_for_random = max_value -1 
        else:
            min_val_for_random = consumo.min()

        random.seed(seed)
        return [random.randint(int(min_val_for_random), max_value) for _ in range(len(consumo))]

    def getDiccionarioAlcaldiasColonias(self):
        if self.data.empty:
            return {}
            
        colonias_list = self.getListaColonias()
        alcaldias_list = self.getListaAlcaldias()
        consumo_total_list = self.getListaConsumoTotal()
        agua_transportada_list = self.generarAguaTransportada()

        if not agua_transportada_list: # Si no se pudo generar agua transportada
            return {}

        dicc = {}
        for i in range(len(alcaldias_list)):
            alcaldia = alcaldias_list[i]
            colonia = colonias_list[i]
            transporte = agua_transportada_list[i]
            consumo = consumo_total_list[i]

            if alcaldia not in dicc:
                dicc[alcaldia] = {}
            if colonia not in dicc[alcaldia]:
                dicc[alcaldia][colonia] = [[], []] # Inicializar listas para transporte y consumo
            
            dicc[alcaldia][colonia][0].append(transporte)
            dicc[alcaldia][colonia][1].append(consumo)
        return dicc

def filtroAlcaldiaColonia(app_instance): # Renombrado para evitar confusión
    st.sidebar.title("Filtros")

    alcaldias_options = ["-- Todas las alcaldías --"] + sorted(list(app_instance.getAlcaldias()))
    alcaldia_seleccionada = st.sidebar.selectbox("Selecciona una alcaldía", alcaldias_options)

    colonia_seleccionada = "-- Todas las colonias --"
    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        if not app_instance.data.empty:
            colonias_filtradas = app_instance.data[app_instance.data["alcaldia"] == alcaldia_seleccionada]["colonia"].unique()
            colonias_options = ["-- Todas las colonias --"] + sorted(list(colonias_filtradas))
            colonia_seleccionada = st.sidebar.selectbox("Selecciona una colonia:", colonias_options)
        else:
            st.sidebar.text("No hay datos de consumo para filtrar colonias.")
    
    return alcaldia_seleccionada, colonia_seleccionada

def mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada):
    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        if colonia_seleccionada == "-- Todas las colonias --":
            st.subheader(f"Datos en {alcaldia_seleccionada}")
        else:
            st.subheader(f"Datos en {alcaldia_seleccionada} para la colonia {colonia_seleccionada}")
    else:
        st.subheader("Datos para toda la CDMX")


# --- INICIO DE LA APP STREAMLIT ---

# URLs o paths a los archivos CSV
consumo_file_path = 'data/consumo_agua_historico_2019.csv'
reportes_2024_file_path = 'data/reportes_agua_2024_01.csv'
reportes_hist_file_path = 'data/reportes_agua_hist.csv'

# Inicializamos la aplicación y cargamos los datos
app_instance = Application(consumo_file_path, reportes_2024_file_path, reportes_hist_file_path)

st.title('Análisis del Consumo y Reportes de Agua en CDMX')

# Limpiamos los DataFrames
app_instance.limpiar_dataframes()

# Obtenemos datos para análisis (basados en consumo)
diccionario_alcaldias_colonias = app_instance.getDiccionarioAlcaldiasColonias()

# Opciones de análisis
tabs_titles = ["Introducción", "Datos de Consumo", "Análisis de Regresión", "Clasificación y Segmentación", "Mapa Interactivo"]
tabs = st.tabs(tabs_titles)

# Filtros en la sidebar
alcaldia_seleccionada, colonia_seleccionada = filtroAlcaldiaColonia(app_instance)

# Tab de Introducción
with tabs[0]:
    st.markdown("""
        ## Análisis Integral del Consumo de Agua en la CDMX
        El agua es un recurso vital y estratégico... (Tu texto de introducción)
        """)

# Tab de Datos
with tabs[1]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)
    
    df_consumo_filtrado = app_instance.data
    if not df_consumo_filtrado.empty:
        if alcaldia_seleccionada != "-- Todas las alcaldías --":
            df_consumo_filtrado = df_consumo_filtrado[df_consumo_filtrado["alcaldia"] == alcaldia_seleccionada]
        if colonia_seleccionada != "-- Todas las colonias --":
            df_consumo_filtrado = df_consumo_filtrado[df_consumo_filtrado["colonia"] == colonia_seleccionada]
        
        st.dataframe(df_consumo_filtrado)
        st.markdown("""
            En esta sección puedes explorar el conjunto de datos de **consumo de agua**. (Descripción de columnas)...
            """)
    else:
        st.warning("No hay datos de consumo disponibles para mostrar.")

# Tab de Análisis de Regresión
with tabs[2]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)
    datos_transporte = []
    datos_consumo = []

    if not diccionario_alcaldias_colonias:
        st.warning("No hay datos procesados para el análisis de regresión.")
    else:
        if alcaldia_seleccionada == "-- Todas las alcaldías --":
            for alcaldia_data in diccionario_alcaldias_colonias.values():
                for colonia_data in alcaldia_data.values():
                    datos_transporte.extend(colonia_data[0])
                    datos_consumo.extend(colonia_data[1])
        elif alcaldia_seleccionada in diccionario_alcaldias_colonias:
            alcaldia_actual = diccionario_alcaldias_colonias[alcaldia_seleccionada]
            if colonia_seleccionada == "-- Todas las colonias --":
                for colonia_data in alcaldia_actual.values():
                    datos_transporte.extend(colonia_data[0])
                    datos_consumo.extend(colonia_data[1])
            elif colonia_seleccionada in alcaldia_actual:
                colonia_actual = alcaldia_actual[colonia_seleccionada]
                datos_transporte.extend(colonia_actual[0])
                datos_consumo.extend(colonia_actual[1])
        
        if len(datos_transporte) > 1 and len(datos_transporte) == len(datos_consumo):
            X = np.array(datos_transporte).reshape(-1, 1)
            Y = np.array(datos_consumo)
            modelo = LinearRegression().fit(X, Y)
            y_pred = modelo.predict(X)
            df_plot = pd.DataFrame({"Agua Transportada": datos_transporte, "Consumo de Agua": datos_consumo, "Predicción de Consumo": y_pred})
            fig = px.scatter(df_plot, x="Agua Transportada", y="Consumo de Agua", color_discrete_sequence=["blue"])
            fig.add_scatter(x=df_plot["Agua Transportada"], y=df_plot["Predicción de Consumo"], mode="lines", name="Línea de Regresión", line=dict(color="red"))
            st.plotly_chart(fig)
        else:
            st.warning("No hay suficientes datos para entrenar el modelo con los filtros seleccionados.")
        st.markdown("### Análisis con Regresión Lineal... (Tu texto explicativo)")

# Tab de Clasificación y Segmentación
with tabs[3]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)
    registros = []

    if not diccionario_alcaldias_colonias:
        st.warning("No hay datos procesados para el análisis de clasificación.")
    else:
        for alcaldia_nombre, alcaldia_data in diccionario_alcaldias_colonias.items():
            if alcaldia_seleccionada != "-- Todas las alcaldías --" and alcaldia_nombre != alcaldia_seleccionada:
                continue
            for colonia_nombre, valores in alcaldia_data.items():
                if colonia_seleccionada != "-- Todas las colonias --" and colonia_nombre != colonia_seleccionada:
                    continue
                transporte_list, consumo_list = valores[0], valores[1]
                for transporte, consumo in zip(transporte_list, consumo_list):
                    diferencia = transporte - consumo
                    promedio = (transporte + consumo) / 2 if (transporte + consumo) > 0 else 0 # Avoid division by zero
                    categoria = "Medio" # Default
                    if diferencia <= 0: categoria = "Peligro"
                    elif promedio > 0 and diferencia >= promedio : categoria = "Perfecto" # Check promedio > 0

                    registros.append({"Agua Transportada": transporte, "Consumo de Agua": consumo, "Categoría": categoria})
        
        if registros:
            df_clasificacion = pd.DataFrame(registros)
            X_clas = df_clasificacion[["Agua Transportada", "Consumo de Agua"]].values
            y_map = {"Peligro": 0, "Medio": 1, "Perfecto": 2}
            y_clas = df_clasificacion["Categoría"].map(y_map).values
            if len(np.unique(y_clas)) > 1 and len(X_clas) >= 5 : # KNN needs at least n_neighbors samples and more than 1 class
                 knn = KNeighborsClassifier(n_neighbors=min(5, len(X_clas))) # Adjust n_neighbors
                 knn.fit(X_clas, y_clas)
                 fig_clas = px.scatter(df_clasificacion, x="Agua Transportada", y="Consumo de Agua", color="Categoría", symbol="Categoría",
                                  labels={"Agua Transportada": "Agua Transportada (L)", "Consumo de Agua": "Consumo de Agua (L)"})
                 st.plotly_chart(fig_clas)
            else:
                st.warning("No hay suficientes datos o diversidad de categorías para el modelo KNN con los filtros seleccionados.")
        else:
            st.warning("No hay datos disponibles para la clasificación con los filtros seleccionados.")
        st.markdown("### Análisis con Modelo de Clasificación... (Tu texto explicativo)")

# Tab de Mapa Interactivo
with tabs[4]:
    mostrarSubtitulo(alcaldia_seleccionada, colonia_seleccionada)
    
    layers = []
    
    # Filtrar datos de consumo para el mapa
    df_map_consumo = app_instance.data.copy()
    if not df_map_consumo.empty:
        if alcaldia_seleccionada != "-- Todas las alcaldías --":
            df_map_consumo = df_map_consumo[df_map_consumo["alcaldia"] == alcaldia_seleccionada]
        if colonia_seleccionada != "-- Todas las colonias --":
            df_map_consumo = df_map_consumo[df_map_consumo["colonia"] == colonia_seleccionada]
        
        if not df_map_consumo.empty:
            consumo_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map_consumo,
                get_position=['longitud', 'latitud'],
                get_fill_color=[0, 120, 255, 140], # Azul para consumo
                get_radius=100, # Radios para diferenciar
                radius_min_pixels=3,
                radius_max_pixels=30,
                pickable=True,
                auto_highlight=True,
                tooltip={
                    "html": "<b>Colonia:</b> {colonia}<br/><b>Alcaldía:</b> {alcaldia}<br/><b>Consumo Total:</b> {consumo_total} L<br/><b>Índice Des.:</b> {indice_des}",
                    "style": {"color": "white", "backgroundColor": "rgba(0,0,0,0.7)", "border": "1px solid white", "padding": "5px"}
                }
            )
            layers.append(consumo_layer)

    # Filtrar datos de reportes 2024 para el mapa
    df_map_reportes_2024 = app_instance.reportes_2024_data.copy()
    if not df_map_reportes_2024.empty:
        if alcaldia_seleccionada != "-- Todas las alcaldías --":
            df_map_reportes_2024 = df_map_reportes_2024[df_map_reportes_2024["alcaldia"] == alcaldia_seleccionada]
        if colonia_seleccionada != "-- Todas las colonias --":
             df_map_reportes_2024 = df_map_reportes_2024[df_map_reportes_2024["colonia"] == colonia_seleccionada]

        if not df_map_reportes_2024.empty:
            reportes_2024_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map_reportes_2024,
                get_position=['longitud', 'latitud'],
                get_fill_color=[255, 0, 0, 180], # Rojo para reportes 2024
                get_radius=70,
                radius_min_pixels=2,
                radius_max_pixels=20,
                pickable=True,
                auto_highlight=True,
                tooltip={
                    "html": "<b>Reporte (2024)</b><br/><b>Folio:</b> {folio_incidente}<br/><b>Clasificación:</b> {clasificacion}<br/><b>Colonia:</b> {colonia}<br/><b>Alcaldía:</b> {alcaldia}",
                    "style": {"color": "white", "backgroundColor": "rgba(0,0,0,0.7)", "border": "1px solid white", "padding": "5px"}
                }
            )
            layers.append(reportes_2024_layer)
            
    # Filtrar datos de reportes históricos para el mapa
    df_map_reportes_hist = app_instance.reportes_hist_data.copy()
    if not df_map_reportes_hist.empty:
        if alcaldia_seleccionada != "-- Todas las alcaldías --":
            df_map_reportes_hist = df_map_reportes_hist[df_map_reportes_hist["alcaldia"] == alcaldia_seleccionada]
        if colonia_seleccionada != "-- Todas las colonias --":
            df_map_reportes_hist = df_map_reportes_hist[df_map_reportes_hist["colonia"] == colonia_seleccionada]

        if not df_map_reportes_hist.empty:
            reportes_hist_layer = pdk.Layer(
                "ScatterplotLayer",
                data=df_map_reportes_hist,
                get_position=['longitud', 'latitud'],
                get_fill_color=[255, 165, 0, 180], # Naranja para reportes históricos
                get_radius=70,
                radius_min_pixels=2,
                radius_max_pixels=20,
                pickable=True,
                auto_highlight=True,
                tooltip={
                    "html": "<b>Reporte Histórico</b><br/><b>Folio:</b> {folio}<br/><b>Tipo Falla:</b> {tipo_de_falla}<br/><b>Colonia:</b> {colonia}<br/><b>Alcaldía:</b> {alcaldia}",
                    "style": {"color": "white", "backgroundColor": "rgba(0,0,0,0.7)", "border": "1px solid white", "padding": "5px"}
                }
            )
            layers.append(reportes_hist_layer)

    if layers:
        # Determinar la vista inicial del mapa
        # Si hay datos filtrados, centrar en ellos, sino en CDMX
        lat_center, lon_center, map_zoom = 19.4326, -99.1332, 10 # CDMX default

        temp_df_for_view = pd.DataFrame()
        if not df_map_consumo.empty : temp_df_for_view = pd.concat([temp_df_for_view, df_map_consumo[['latitud', 'longitud']]])
        if not df_map_reportes_2024.empty : temp_df_for_view = pd.concat([temp_df_for_view, df_map_reportes_2024[['latitud', 'longitud']]])
        if not df_map_reportes_hist.empty : temp_df_for_view = pd.concat([temp_df_for_view, df_map_reportes_hist[['latitud', 'longitud']]])
        
        if not temp_df_for_view.empty:
            lat_center = temp_df_for_view['latitud'].astype(float).mean()
            lon_center = temp_df_for_view['longitud'].astype(float).mean()
            if alcaldia_seleccionada != "-- Todas las alcaldías --": map_zoom = 12
            if colonia_seleccionada != "-- Todas las colonias --": map_zoom = 14
            
        view_state = pdk.ViewState(
            latitude=lat_center,
            longitude=lon_center,
            zoom=map_zoom,
            pitch=45, # Angulo para mejor visualización 3D
        )
        
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v10', # Estilo de mapa
            initial_view_state=view_state,
            layers=layers,
            api_keys={'mapbox': 'YOUR_MAPBOX_API_KEY'} # Reemplaza con tu API Key de Mapbox si usas estilos que la requieran
        ))
        st.markdown("""
            Leyenda del Mapa:
            - **Puntos Azules**: Consumo de agua (mayor radio).
            - **Puntos Rojos**: Reportes de agua recientes (2024).
            - **Puntos Naranjas**: Reportes de agua históricos.
            
            Pasa el cursor sobre los puntos para más detalles.
            """)
    else:
        st.warning("No hay datos disponibles para mostrar en el mapa con los filtros seleccionados.")

    st.markdown("### Mapa Interactivo de Consumo y Reportes de Agua... (Tu texto explicativo)")