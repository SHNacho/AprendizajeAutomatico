# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: 
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
#from sympy import diff, exp
#from sympy.abc import u,v

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# =============================================================================
# Ejercicio 1.1
# =============================================================================
def gradient_descent(w_ini, lr, grad_fun, fun, epsilon = None, max_iters = 50):
    #
    # gradiente descendente
    # 
    w = w_ini
    ws = np.array(w)
    iterations = 0
    continuar = True
    while(continuar and iterations < max_iters):
        w = w - lr * grad_fun(w[0], w[1])
        iterations += 1
        ws = np.append(ws, w, axis=0)
        if epsilon != None:
            continuar = fun(w[0], w[1]) > epsilon
        
    ws = np.reshape(ws, (int(len(ws)/2), 2))
    return w, iterations, ws   

# =============================================================================
# Ejercicio 1.2
# =============================================================================
def E(u,v):
    return (u*v*np.exp(-(u**2) -(v**2)))**2  

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return -4*u**3*v**2*np.exp(-2*u**2 - 2*v**2) + 2*u*v**2*np.exp(-2*u**2 - 2*v**2)
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return -4*u**2*v**3*np.exp(-2*u**2 - 2*v**2) + 2*u**2*v*np.exp(-2*u**2 - 2*v**2)

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


eta = 0.1 
maxIter = 10000000000
error2get = 1e-8
initial_point = np.array([0.5,-0.5])
w, it, ws = gradient_descent(initial_point, 0.1, gradE, E, error2get, maxIter)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


'''
Esta función muestra una figura 3D con la función a optimizar junto con el 
óptimo encontrado y la ruta seguida durante la optimización. Esta función, al igual
que las otras incluidas en este documento, sirven solamente como referencia y
apoyo a los estudiantes. No es obligatorio emplearlas, y pueden ser modificadas
como se prefiera. 
    rng_val: rango de valores a muestrear en np.linspace()
    fun: función a optimizar y mostrar
    ws: conjunto de pesos (pares de valores [x,y] que va recorriendo el optimizador
                           en su búsqueda iterativa del óptimo)
    colormap: mapa de color empleado en la visualización
    title_fig: título superior de la figura
    
Ejemplo de uso: display_figure(2, E, ws, 'plasma','Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
'''
def display_figure(rng_val, fun, ws, colormap, title_fig):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    #from mpl_toolkits.mplot3d import Axes3D
    X = np.linspace(-rng_val, rng_val, 50)
    Y = np.linspace(-rng_val, rng_val, 50)
    X, Y = np.meshgrid(X, Y)
    Z = fun(X, Y) 
    fig = plt.figure(figsize=(10, 10))
    #ax = Axes3D(fig,auto_add_to_figure=False)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    #fig.add_axes(ax)
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                            cstride=1, cmap=colormap, alpha=.6)
    if len(ws)>0:
        ws = np.asarray(ws)
        min_point = np.array([ws[-1,0],ws[-1,1]])
        min_point_ = min_point[:, np.newaxis]
        ax.plot(ws[:-1,0], ws[:-1,1], E(ws[:-1,0], ws[:-1,1]), 'r*', markersize=5)
        ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('E(u,v)')
    plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


display_figure(1, E, ws, 'viridis', 'Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')

#%%
# =============================================================================
# Ejercicio 1.3
# =============================================================================

def f(x, y):
    return x**2 + 2 * (y**2) + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)

def d_fx(x, y):
    a = np.pi * y
    return 4 * np.pi * np.sin(np.pi * y) * np.cos(2 * np.pi * x) + 2 * x

def d_fy(x, y):
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(np.pi * y) + 4 * y

def grad_f(x, y):
    return np.array([d_fx(x,y), d_fy(x,y)])
                    
lr = 0.01
maxIter = 50
#error2get = 1e-8
initial_point = np.array([-1.0,1.0])
w, it, ws = gradient_descent(initial_point, lr, grad_f, f, max_iters=maxIter)

print(f(w[0], w[1]), it)

#%%






###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa	
def pseudoinverse(?):
    #
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 

#Seguir haciendo el ejercicio...



