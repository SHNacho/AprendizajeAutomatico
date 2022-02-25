# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
import numpy as np
# Cargamos la base de datos de Iris
iris = load_iris()
# Imprimimos las claves del diccionario contiene la base de datos
caracteristicas = iris['data']
clase = iris['target']
# Insertamos la clase dentro del array de características para tenerlo 
# todo en un mismo array
data = np.insert(caracteristicas, 0, clase, 1)

# Nos quedamos con las características primera y tercera y con el elemento
# en la primera posición que es la clase
# posición:
# 0 -> Clase
# 1 -> Primera característica
# 3 -> Tercera característica
data = data[ : , [0, 1, 3]]


