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
    page_icon = 'gota.jpg',
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
            [Repositorio de Girhub](https://github.com/JoseManuelAlonsoMorales/FinalProjectAI)
        """
    }
)

class Application:
    def __init__(self):
        self.data = None
        self.getData()

    def getData(self):
        self.data = pd.read_csv('data/consumo_agua_historico_2019.csv')

app = Application()

st.title('Proyecto Final IA')

st.write('hola mundo')
