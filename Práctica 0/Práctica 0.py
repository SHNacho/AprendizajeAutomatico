# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Ejercicio 1
# =============================================================================

# Cargamos la base de datos de Iris
iris = load_iris()

#print(iris.keys())

# Obtenemos las características
caracteristicas = iris['data']
# Obtenemos las clases
clases = iris['target']
# Obtenemos el nombre de las clases y de las características
target_names = iris['target_names']
feature_names = iris['feature_names']

# Nos quedamos con las características primera y tercera de cada fila
caracteristicas = caracteristicas[ : , [0, 2]]

# Insertamos la clase dentro del array de características para tenerlo 
# todo en un mismo array
data = np.insert(caracteristicas, 0, clases, 1)

# Separamos las características dependiendo de cada clase
c_setosa = data[data[:, 0] == 0., : ]
c_versicolor = data[data[:, 0] == 1., : ]
c_virginica = data[data[:, 0] == 2., : ]




# Pintamos las caracterísiticas en un Scatter Plot.
plt.scatter(c_setosa[:, 1], c_setosa[:, 2], c='red', label=target_names[0])
plt.scatter(c_versicolor[:, 1], c_versicolor[:, 2], c='green', label=target_names[1])
plt.scatter(c_virginica[:, 1], c_virginica[:, 2], c='blue', label=target_names[2])
# Le ponemos nombre a los ejes
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
# Pintamos la leyenda
plt.legend()
plt.show()

# =============================================================================
# Ejercicio 2
# =============================================================================
# Mezclamos aleatoriamente los vectores de datos
# np.random.shuffle(c_setosa)
# np.random.shuffle(c_versicolor)
# np.random.shuffle(c_virginica)

np.random.shuffle(data)

# Calculamos el tamaño de las particiones
training_split = int(np.rint(len(c_setosa)*0.8))
print(training_split)

# La partición de training estará en el rango [0, training_split)
# setosa_training = [ : training_split, :]
# versicolor_training = c_setosa[ : training_split, :]
# virginica_training = c_setosa[ : training_split, :]
setosa_data = np.where(data[:, 0] == 0.)[0]
versicolor_data = np.where(data[:, 0] == 1.)[0]
virginica_data = np.where(data[:, 0] == 2.)[0]

setosa_training = setosa_data[ : training_split]
versicolor_training = versicolor_data[ : training_split]
virginica_training = virginica_data[ : training_split]
setosa_test = setosa_data[ training_split : ]
versicolor_test = versicolor_data[ training_split : ]
virginica_test = virginica_data[training_split : ]

# Unimos los arrays de training en uno solo
training_data = np.concatenate((setosa_training, versicolor_training, virginica_training))
training_data = np.sort(training_data)


# Unimos los arrays de test en uno solo
test_data = np.concatenate((setosa_test, versicolor_test, virginica_test))
test_data = np.sort(test_data)


# Mostramos la salida
print("--- Clase setosa ---")
print("Ejemplos training:  ", len(setosa_training))
print("Ejemplos test:      ", len(setosa_test))
print("--- Clase versicolor ---")
print("Ejemplos training:  ", len(versicolor_training))
print("Ejemplos test:      ", len(versicolor_test))
print("--- Clase virginica ---")
print("Ejemplos training:  ", len(virginica_training))
print("Ejemplos test:      ", len(virginica_test))
print("Clase de los ejemplos de entrenamiento:")
print(data[training_data, 0])
print("Clase de los ejemplos de test:")
print(data[test_data, 0])

# =============================================================================
# Ejercicio 3
# =============================================================================
vals = np.linspace(0, np.pi, num=100)













