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

class Application:
    def __init__(self):
        self.data = None
        self.getData()

    def getData(self):
        url = "https://raw.githubusercontent.com/JoseManuelAlonsoMorales/FinalProjectAI/main/data/consumo_agua_historico_2019.csv"
        self.data = pd.read_csv(url)


app = Application()
df = app.data

st.title('Proyecto Final IA')

listaConsumoTotal = df["consumo_total"].tolist() # Vaiable Independiente, Lista del consumo total de cada delegacion

listaColonias = df["colonia"].tolist() #Lista colonias
listaAlcaldias = df['alcaldia'].tolist() #Lista Alcaldias

cantConsumidaMaxMin = np.array(listaConsumoTotal) #Convertir el consumo total para poder ver el min y max y poder crear los datos de agua transportada

print(cantConsumidaMaxMin.max())
print(cantConsumidaMaxMin.min())

random.seed(2004)
listaAguaTransortada = [] #Lista del agua transportada
for i in range(len(listaColonias)): #Ciclo para meter los datos creados a la lista anterior
    AguaTransportada = random.randint(int(cantConsumidaMaxMin.min()), 15000)
    listaAguaTransortada.append(AguaTransportada)

diccionarioAlcaldias_Colonias = {} #Diccionario de las colonias separadas por alcadias, junto con sus datos de obtencion y consumo de agua

for i in range(len(listaAlcaldias)):
    alcaldia = listaAlcaldias[i]
    colonia = listaColonias[i]
    transporte = listaAguaTransortada[i]
    consumo = listaConsumoTotal[i]

    if alcaldia not in diccionarioAlcaldias_Colonias:
        diccionarioAlcaldias_Colonias[alcaldia] = {}
    if colonia in diccionarioAlcaldias_Colonias[alcaldia]:
        diccionarioAlcaldias_Colonias[alcaldia][colonia][0].append(transporte)
        diccionarioAlcaldias_Colonias[alcaldia][colonia][1].append(consumo)
    else:
        diccionarioAlcaldias_Colonias[alcaldia][colonia] = [[transporte], [consumo]]

# Sidebar
st.sidebar.title("Opciones de análisis")
opcion = st.sidebar.selectbox(
    "Selecciona el modelo a aplicar:",
    ("Ver Dataframe", "Regresión Lineal", "Clasificación")
)

if opcion == "Ver Dataframe":
    dataSetAgua = pd.read_csv("consumo_agua_historico_2019.csv")
    dataSetAgua.describe()

if opcion == "Regresión Lineal":
    st.subheader("Modelo de Regresión Lineal")

    datos = diccionarioAlcaldias_Colonias['BENITO JUAREZ']
    for colonia in datos:
        if colonia == "MODERNA":
            datosTransporte = diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][0]
            datosConsumo = diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][1]

    print(datosConsumo)
    print(datosTransporte)

    X = np.array([datosTransporte]).reshape(-1,1)
    Y = np.array(datosConsumo)

    modelo = LinearRegression()

    # Entrenar el modelo con los datos
    modelo.fit(X, Y)

    # Realizar predicciones
    y_pred = modelo.predict(X)

    # Visualizar los resultados
    plt.scatter(X, Y, color='blue') 
    plt.scatter(X, y_pred, color='green') # Datos originales
    plt.plot(X, y_pred, color='red')  
    plt.ylim(-5,3000)
    plt.xlabel('Variable independiente')
    plt.ylabel('Variable dependiente')
    plt.title('Regresión lineal')
    plt.show()

if opcion == "Clasificación":
    st.subheader("Modelo de Clasificación")

    #Clasificacion
    ArrayDatos = []
    DatosY = []
    for i in range(len(diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][0])):
        DatosNuevosArray = [diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][0][i], diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][1][i]]
        ArrayDatos.append(DatosNuevosArray)

        Diferencia = diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][0][i] - diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][1][i]
        Prom = (diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][0][i] + diccionarioAlcaldias_Colonias['BENITO JUAREZ']['MODERNA'][1][i])/2
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
