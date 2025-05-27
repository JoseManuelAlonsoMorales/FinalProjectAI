import streamlit as st
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import toml
from matplotlib.colors import ListedColormap
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

import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

if opcion == "Regresión Lineal":
    st.subheader("Modelo de Regresión Lineal")

    # Filtros
    alcaldias = ["-- Todas las alcaldías --"] + sorted(diccionario_alcaldias_colonias.keys())
    alcaldia_seleccionada = st.selectbox("Selecciona una alcaldía:", alcaldias)

    colonia_seleccionada = "-- Todas las colonias --"
    if alcaldia_seleccionada != "-- Todas las alcaldías --":
        colonias = ["-- Todas las colonias --"] + sorted(diccionario_alcaldias_colonias[alcaldia_seleccionada].keys())
        colonia_seleccionada = st.selectbox("Selecciona una colonia:", colonias)

    # Recolectar datos según filtros
    datosTransporte = []
    datosConsumo = []

    if alcaldia_seleccionada == "-- Todas las alcaldías --":
        # Agregamos todos los datos de todas las alcaldías y colonias
        for alcaldia in diccionario_alcaldias_colonias:
            for colonia in diccionario_alcaldias_colonias[alcaldia]:
                datosTransporte.extend(diccionario_alcaldias_colonias[alcaldia][colonia][0])
                datosConsumo.extend(diccionario_alcaldias_colonias[alcaldia][colonia][1])
        titulo = "Regresión lineal: todas las alcaldías y colonias"
        
    elif colonia_seleccionada == "-- Todas las colonias --":
        for colonia in diccionario_alcaldias_colonias[alcaldia_seleccionada]:
            datosTransporte.extend(diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia][0])
            datosConsumo.extend(diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia][1])
        titulo = f"Regresión lineal: {alcaldia_seleccionada} (todas sus colonias)"
        
    else:
        datosTransporte = diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia_seleccionada][0]
        datosConsumo = diccionario_alcaldias_colonias[alcaldia_seleccionada][colonia_seleccionada][1]
        titulo = f"Regresión lineal: {colonia_seleccionada}, {alcaldia_seleccionada}"

    # Validar datos
    if len(datosTransporte) > 1 and len(datosTransporte) == len(datosConsumo):
        X = np.array(datosTransporte).reshape(-1, 1)
        Y = np.array(datosConsumo)

        modelo = LinearRegression()
        modelo.fit(X, Y)
        y_pred = modelo.predict(X)

        df_plot = pd.DataFrame({
            "Agua Transportada": datosTransporte,
            "Consumo de Agua": datosConsumo,
            "Predicción de Consumo": y_pred
        })

        fig = px.scatter(df_plot, x="Agua Transportada", y="Consumo de Agua", 
                         color_discrete_sequence=["blue"], title=titulo)
        fig.add_scatter(x=df_plot["Agua Transportada"], y=df_plot["Predicción de Consumo"], 
                        mode="lines", name="Línea de Regresión", line=dict(color="red"))
        st.plotly_chart(fig)
    else:
        st.warning("No hay suficientes datos para entrenar el modelo.")

if opcion == "Clasificación":
    st.subheader("Modelo de Clasificación")

    #Clasificacion
    ArrayDatos = []
    DatosY = []
    for i in range(len(diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][0])):
        DatosNuevosArray = [diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][0][i], diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][1][i]]
        ArrayDatos.append(DatosNuevosArray)

        Diferencia = diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][0][i] - diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][1][i]
        Prom = (diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][0][i] + diccionario_alcaldias_colonias['BENITO JUAREZ']['MODERNA'][1][i])/2
        if Diferencia <= 0:
            DatosY.append(0)
        elif Diferencia >= Prom:
            DatosY.append(2)
        elif Diferencia < Prom:
            DatosY.append(1)


    Xclasificacion = np.array(ArrayDatos)
    YClasificacion = np.array(DatosY)

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(Xclasificacion, YClasificacion)

    plt.figure(figsize=(8, 6))

    # Graficar los datos de entrenamiento
    plt.scatter(Xclasificacion[YClasificacion == 0][:, 0], Xclasificacion[YClasificacion == 0][:, 1], color='red', label='Peligro', marker='x')
    plt.scatter(Xclasificacion[YClasificacion == 1][:, 0], Xclasificacion[YClasificacion == 1][:, 1], color='orange', label='Medio', marker='o')
    plt.scatter(Xclasificacion[YClasificacion == 2][:, 0], Xclasificacion[YClasificacion == 2][:, 1], color='green', label='Perfecto', marker='d')

    plt.xlabel('Velocidad')
    plt.ylabel('Manejo')
    plt.title('Clasificación de Personajes Mario Kart según estadísticas')
    plt.legend()
    plt.grid(True)
    plt.show()
