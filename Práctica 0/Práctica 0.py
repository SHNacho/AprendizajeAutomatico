# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
np.random.shuffle(data)

# Calculamos el tamaño de las particiones
training_split = int(np.rint(len(c_setosa)*0.8))


# Nos quedamos con los índices de los datos de cada clase
setosa_data = np.where(data[:, 0] == 0.)[0]
versicolor_data = np.where(data[:, 0] == 1.)[0]
virginica_data = np.where(data[:, 0] == 2.)[0]

# Separamos los arrays de índices anteriores en training y test
# La partición de training estará en el rango [0, training_split)
setosa_training = setosa_data[ : training_split]
versicolor_training = versicolor_data[ : training_split]
virginica_training = virginica_data[ : training_split]
# La partición de training estará en el rango [training_split, fin)
setosa_test = setosa_data[ training_split : ]
versicolor_test = versicolor_data[ training_split : ]
virginica_test = virginica_data[training_split : ]

# Unimos los arrays de training en uno solo
training_data = np.concatenate((setosa_training, versicolor_training, virginica_training))
# Ordenamos los índices
training_data = np.sort(training_data)

# Unimos los arrays de test en uno solo
test_data = np.concatenate((setosa_test, versicolor_test, virginica_test))
# Ordenamos los índices
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
vals = np.linspace(0, 4*np.pi, num=100)
f1 = lambda x : 10**(-5) * np.sinh(x)
f2 = lambda x : np.cos(x)
f3 = lambda x : np.tanh(2 * np.sin(x) - 4 * np.cos(x))

f1_vals = f1(vals)
f2_vals = f2(vals)
f3_vals = f3(vals)

plt.plot(vals, f1_vals, 'g--', label='y = 1e-5*sinh(x)')
plt.plot(vals, f2_vals, 'k--', label='y = cos(x)')
plt.plot(vals, f3_vals, 'r--', label='y = tanh(2*sin(x)-4*cos(x))')

plt.legend(loc='upper left')
plt.show()

# =============================================================================
# Ejercicio 4
# =============================================================================
# Definimos las dos funciones
f1 = lambda x, y: 1 - np.abs(x + y) - np.abs(y - x)
f2 = lambda x, y: x * y * np.exp(-(x**2)-(y**2))

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=(20, 20))

# =============
# Primer subplot
# =============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
# Le ponemos título a la figura con un tamaño de fuente 20
ax.set_title('Pirámide', fontsize=20)

# Generamos los valores de x e y
X = np.arange(-6, 6.4, 0.4)
Y = np.arange(-6, 6.4, 0.4)
# Obtenemos las matrices de coordenadas a partir de X e Y
X, Y = np.meshgrid(X, Y)
# Evaluamos los valores
Z = f1(X, Y)
# Dibujamos la gráfica con los valores de X, Y, Z, con un mapa de color 
# 'coolwarm' y con antialiased
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=True)
# Establecemos los valores de los ejes para que sean como los del pdf
ax.set_zlim(-11, 0.5)
ax.xaxis.set_ticks(np.arange(-6, 7, 2))
ax.yaxis.set_ticks(np.arange(-6, 7, 2))

# ==============
# Segundo subplot
# ==============
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
# Le ponemos título a la figura
ax.set_title('$x \cdot y \cdot e^{(-x^2 -y^2)}$', fontsize=20)
# Generamos los valores de x e y
X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)
# Obtenemos las matrices de coordenadas a partir de X e Y
X, Y = np.meshgrid(X, Y)
# Evaluamos los valores
Z = f2(X, Y)
# Dibujamos la gráfica con los valores de X, Y, Z, con un mapa de color 
# 'viridis' y con antialiased
ax.plot_surface(X, Y, Z,rstride=1, cstride=1, cmap=cm.viridis, antialiased=True)
# Establecemos los valores de los ejes para que sean como los del pdf
ax.set_zlim(-0.18, 0.18)
ax.xaxis.set_ticks(np.arange(-2, 2.5, 0.5))
ax.yaxis.set_ticks(np.arange(-2, 2.5, 0.5))

plt.show()