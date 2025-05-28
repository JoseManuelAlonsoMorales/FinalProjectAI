import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import random
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

# Configuración de la página
st.set_page_config(
    page_title = 'Proyecto Final IA',
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


# Inicializamos la aplicación y cargamos los datos
app = Application(r'https://raw.githubusercontent.com/JoseManuelAlonsoMorales/FinalProjectAI/main/data/consumo_agua_historico_2019.csv')  # URL del archivo CSV
df = app.getDataFrame()

st.title('Proyecto Final IA')

# Comenzamos a trabajar con los datos
app.limpiarDataFrame()  # Limpiamos el DataFrame de valores nulos y cadenas vacías

cant_consumida_max_min = np.array(app.getListaConsumoTotal()) # Convertimos la lista de consumo total a un array de numpy para obtener los valores máximos y mínimos
diccionario_alcaldias_colonias = app.getDiccionarioAlcaldiasColonias() # Obtenemos el diccionario de alcaldías y colonias con los datos de transporte y consumo

# Sidebar con las opciones de análisis
st.sidebar.title("Opciones de análisis")
opcion = st.sidebar.selectbox(
    "Selecciona el modelo a aplicar:",
    ("Ver Dataframe", "Regresión Lineal", "Clasificación")
)

# Si el usuario selecciona "Ver Dataframe", mostramos el DataFrame
if opcion == "Ver Dataframe":
    st.subheader("Ver Dataframe")

    alcaldias = sorted(list(app.getAlcaldias()))
    alcaldias = ["-- Todas las alcaldías --"] + alcaldias
    alcaldia_seleccionada = st.selectbox("Selecciona una alcaldía", alcaldias)

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        colonias_filtradas = app.data[app.data["alcaldia"] == alcaldia_seleccionada]["colonia"].unique()
        colonias = sorted(colonias_filtradas)
        colonias = ["-- Todas las colonias --"] + colonias
        colonia_seleccionada = st.selectbox("Selecciona una colonia:", colonias)
    else:
        colonia_seleccionada = "-- Todas las colonias --"

    # Aplicar filtros según selección
    df_filtrado = app.data

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        df_filtrado = df_filtrado[df_filtrado["alcaldia"] == alcaldia_seleccionada]

    if colonia_seleccionada != "-- Todas las colonias --":
        df_filtrado = df_filtrado[df_filtrado["colonia"] == colonia_seleccionada]

    st.subheader(f"Datos{' para ' + colonia_seleccionada if colonia_seleccionada != '-- Todas las colonias --' else ''}"
                 f"{' en ' + alcaldia_seleccionada if alcaldia_seleccionada != '-- Todas las alcaldías --' else ''}")
    
    st.dataframe(df_filtrado)

# Si el usuario selecciona "Regresión Lineal", mostramos el modelo de regresión lineal
if opcion == "Regresión Lineal":
    st.subheader("Modelo de Regresión Lineal")

    alcaldias = sorted(list(app.getAlcaldias()))
    alcaldias = ["-- Todas las alcaldías --"] + alcaldias
    alcaldia_seleccionada = st.selectbox("Selecciona una alcaldía", alcaldias)

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        colonias_filtradas = app.data[app.data["alcaldia"] == alcaldia_seleccionada]["colonia"].unique()
        colonias = sorted(colonias_filtradas)
        colonias = ["-- Todas las colonias --"] + colonias
        colonia_seleccionada = st.selectbox("Selecciona una colonia:", colonias)
    else:
        colonia_seleccionada = "-- Todas las colonias --"

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

    st.subheader(f"Datos{' para ' + colonia_seleccionada if colonia_seleccionada != '-- Todas las colonias --' else ''}"
                 f"{' en ' + alcaldia_seleccionada if alcaldia_seleccionada != '-- Todas las alcaldías --' else ''}")

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

# Si el usuario selecciona "Clasificación", mostramos el modelo de clasificación
if opcion == "Clasificación":
    st.subheader("Modelo de Clasificación")

    alcaldias = sorted(list(app.getAlcaldias()))
    alcaldias = ["-- Todas las alcaldías --"] + alcaldias
    alcaldia_seleccionada = st.selectbox("Selecciona una alcaldía", alcaldias)

    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        colonias_filtradas = app.data[app.data["alcaldia"] == alcaldia_seleccionada]["colonia"].unique()
        colonias = sorted(colonias_filtradas)
        colonias = ["-- Todas las colonias --"] + colonias
        colonia_seleccionada = st.selectbox("Selecciona una colonia:", colonias)
    else:
        colonia_seleccionada = "-- Todas las colonias --"

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
    
    st.subheader(f"Datos{' para ' + colonia_seleccionada if colonia_seleccionada != '-- Todas las colonias --' else ''}"
    f"{' en ' + alcaldia_seleccionada if alcaldia_seleccionada != '-- Todas las alcaldías --' else ''}")

    # Verificación
    if not registros:
        st.warning("No hay datos disponibles para la selección.")
    else:
        df_clasificacion = pd.DataFrame(registros)

        # Entrenamiento del modelo
        X = df_clasificacion[["Agua Transportada", "Consumo de Agua"]].values
        y_map = {"Peligro": 0, "Medio": 1, "Perfecto": 2}
        y = df_clasificacion["Categoría"].map(y_map).values

        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(X, y)

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
