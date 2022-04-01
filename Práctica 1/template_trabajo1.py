# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Ignacio Sánchez Herrera
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

# =============================================================================
# Ejercicio 1.1
# Función que calcula el descenso del gradiente. Devuelve el punt, el número de iteraciones realizadas y el conjunto de puntos por
# los que ha pasado.
# =============================================================================
'''
    Función que calcula el descenso del gradiente de una función.
        w_ini: valor inicial de las variables de la función
        lr: learning rate.
        grad_fun: gradiente de la función a minimizar
        fun: función a minimizar
        epsilon: mínimo deseado. Una vez se alcanza el mínimo, para la búsqueda
        max_iters: máximo número de iteraciones en caso de que no llegue a epsilon
    
    Devuelve:
        w: Punto en el que se alcanza el mínimo
        iterations: número de iteraciones realizadas
        ws: Puntos por los que pasa durante la búsqueda del mínimo
'''
def gradient_descent(w_ini, lr, grad_fun, fun, epsilon = None, max_iters = 50):
    #
    # gradiente descendente
    # 
    w = w_min = w_ini
    ws = np.array([w]) # Array de puntos por los que pasa el algoritmo
    iterations = 0
    continuar = True # Será False cuando alcance epsilon
    # Mientras no se haya alcanzado epsilon y no se hayan realizado max_iters
    while(continuar and iterations < max_iters):
        # Calculamos el nuevo punto mediante la fórmula de descenso del gradiente
        w = w - lr * grad_fun(w[0], w[1])
        iterations += 1 #Incrementa el número de iteraciones
        # Se añade al vector de puntos el nuevo punto
        ws = np.append(ws, [w], axis=0)
        # Actualizamos el mínimo
        if(fun(w_min[0], w_min[1]) > fun(w[0], w[1])):
            w_min = w
        # Comprobamos si se alcanza epsilon
        if epsilon != None:
            continuar = fun(w[0], w[1]) > epsilon
    
    w = w_min
    
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


eta = 0.1       # Learning rate
maxIter = 10000000000 # Máximo número de iteraciones
error2get = 1e-8 # Mínimo a alcanzar
initial_point = np.array([0.5,-0.5]) # Punto inicial
# Calculamos el descenso del gradiente
# w: punto en el que se alcanza el mínimo
# it: Número de iteraciones necesarias
# ws: puntos por los que pasa durante el descenso del gradiente
w, it, ws = gradient_descent(initial_point, eta, gradE, E, error2get, maxIter)

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
        ax.plot(ws[:-1,0], ws[:-1,1], fun(ws[:-1,0], ws[:-1,1]), 'r*', markersize=5)
        ax.plot(min_point_[0], min_point_[1], fun(min_point_[0], min_point_[1]), 'r*', markersize=10)
    if len(title_fig)>0:
        fig.suptitle(title_fig, fontsize=16)
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_zlabel('E(u,v)')
    plt.show()

# input("\n--- Pulsar tecla para continuar ---\n")


# display_figure(1, E, ws, 'viridis', 'Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')

#%%
# =============================================================================
# Ejercicio 1.3
# =============================================================================

def f(x, y):
    return x**2 + 2 * (y**2) + 2 * np.sin(2 * np.pi * x) * np.sin(np.pi * y)

def d_fx(x, y):
    return 4 * np.pi * np.sin(np.pi * y) * np.cos(2 * np.pi * x) + 2 * x

def d_fy(x, y):
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(np.pi * y) + 4 * y

def grad_f(x, y):
    return np.array([d_fx(x,y), d_fy(x,y)])
  
## Apartado a)  
### Con learning rate 0.01        
lr = 0.01
maxIter = 50    # Número máximo de iteraciones
initial_point = np.array([-1.0,1.0]) # Punto inicial
# Calculamos el descenso del gradiente
w, it, ws = gradient_descent(initial_point, lr, grad_f, f, max_iters=maxIter)
# display_figure(2, f, ws, 'viridis', 'Ejercicio 1.3.a. Descenso del gradiente con lr=0.01')

# Mostramos los resultados
print("Punto inicial: ", initial_point)
print("Mínimo alcanzado: ", f(w[0], w[1]))
print("Alcanzado en el punto: (", w[0], ', ', w[1],')')

x_plot = np.arange(0, 51)
y_plot = np.array([f(v[0], v[1]) for v in ws])

# Pintamos la evolución del valor de la función
plt.plot(x_plot, y_plot)
plt.title("Valor de la función $f$ en cada iteración. \n Learning rate 0.01")
plt.show()

### Con learning rate 0.1
lr = 0.1
w, it, ws = gradient_descent(initial_point, lr, grad_f, f, max_iters=maxIter)
# display_figure(3, f, ws, 'viridis', 'Ejercicio 1.3.a. Descenso del gradiente con lr=0.1')

print("Punto inicial: ", initial_point)
print("Mínimo alcanzado: ", f(w[0], w[1]))
print("Alcanzado en el punto: (", w[0], ', ', w[1],')')

# Eje x de 0 a 51. 
# Tenemos 51 valores en ws: las 50 iteraciones más el valor inicial
x_plot = np.arange(0, 51) 
y_plot = np.array([f(v[0], v[1]) for v in ws])

# Pintamos la evolución del valor de la función
plt.plot(x_plot, y_plot)
plt.title("Valor de la función $f$ en cada iteración. \n Learning rate 0.1")
plt.show()

#%%
## Apartado b)

# Definimos los puntos iniciales
initial_points = np.array([[-0.5, -0.5],
                           [ 1.0,  1.0],
                           [ 2.1, -2.1],
                           [-3.0,  3.0],
                           [-2.0,  2.0]])

# Definimos los learning rates
lrs = np.array([0.01, 0.1])

print("Ejercicio 1.3.b")
# Para cada learning rate
for lr in lrs:
    print("============ Learning rate = ", lr, " ==============")
    # Para cada punto inicial
    for ini_p in initial_points:
        # Calculamos descenso del gradiente
        w, it, ws = gradient_descent(ini_p,
                                     lr,
                                     grad_f,
                                     f,
                                     max_iters=maxIter)
        # Mostramos la salida
        print("-----")
        print("Punto inicial: ", ini_p)
        print("Mínimo alcanzado: ", f(w[0], w[1]))
        print("Alcanzado en el punto: (", w[0], ', ', w[1],')')
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
    w_t = np.transpose(w)
    h_x = np.array([np.dot(w_t, x_n) for x_n in x])
    error = h_x - y
    cuadratic_error = error * error
    ecm = np.mean(cuadratic_error)
    
    return ecm

def grad_Err(x, y, w):
    w_t = np.transpose(w)
    n_caracteristicas = x.shape[1]
    d_w = [np.mean((np.array([np.dot(w_t, x_n) for x_n in x]) - y) * x[:, i]) * 2 for i in range(n_caracteristicas)]
    
    return np.array(d_w)

# Gradiente Descendente Estocastico
def sgd(x, y, lr, grad_fun, fun, epsilon = None, epochs = 50, batch_size = 32):
    tam_v = x.shape[1]
    w = w_min = np.zeros(tam_v)
    ws = np.array(w)
    continuar = True
    iterations = 0
    while( continuar and iterations < epochs ):
        # print("Epoch: ", iterations)
        # Obtenemos una permutación aleatoria de 0 a N donde N es el número de 
        # datos, de manera que usaremos la misma permutación para x e y.
        # De esta forma mezclamos los dos vectores de la misma manera.
        permutation = np.random.permutation(len(x))
        # Mezclamos los datos
        x_batches = x[permutation]
        y_batches = y[permutation]
        
        for i in range(0, len(x_batches), batch_size):
             
            w = w - lr * grad_fun(x_batches[i : i + batch_size], y_batches[i : i + batch_size], w)
            ws = np.append(ws, w, axis=0)
            #print("Error: ", fun(x, y, w))
            
            if(fun(x, y, w_min) > fun(x, y, w)):
                w_min = w
            
            if epsilon != None:
                continuar = fun(x, y, w) > epsilon
             

        iterations += 1  
        
    ws = np.reshape(ws, (int(len(ws)/tam_v), tam_v))
    w = w_min
    return w, iterations, ws  

# Pseudoinversa	
def pseudoinverse(X, y):
    #
    U, D, V_t = np.linalg.svd(X, full_matrices = False)
    D = np.diag(D)
    V = np.transpose(V_t)
    
    pseudoinverse = np.dot(V, np.linalg.inv(D))
    pseudoinverse = np.dot(pseudoinverse, np.transpose(U))
    
    w = np.dot(pseudoinverse, y)
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

#%%

w, _, _ = sgd(x, y,
              lr = 0.01,
              grad_fun = grad_Err,
              fun = Err,
              epsilon = 1e-8,
              epochs = 200,
              batch_size = 32)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

#%%

w = pseudoinverse(x, y)

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))
print(w)
#%%
data_1 = np.array([x_i for x_i, y_i in zip(x, y) if y_i == -1])
data_5 = np.array([x_i for x_i, y_i in zip(x, y) if y_i == 1])

hiperplano = lambda x: -(w[1]*x / w[2]) - (w[0]/w[2])

h = [hiperplano(0.), hiperplano(1.)]

plt.scatter(data_1[:, 1], data_1[:, 2], c='blue')
plt.scatter(data_5[:, 1], data_5[:, 2], c='red')
plt.plot(h)
plt.xlabel("Intensidad media")
plt.ylabel("Simetria")
plt.show()

#%%
#input("\n--- Pulsar tecla para continuar ---\n")

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

# Generamos 1000 valores aleatorios en un cuadrado de 1x1
N = 1000
size = 1
d = 2
x = simula_unif(N, d, size)

plt.scatter(x[:, 0], x[:, 1])
plt.title("Mapa de puntos 2D genereados aleatoriamente en un cuadrado de 1x1")
plt.show()
# Obtenemos la etiqueta de cada elemento con la función f
y = np.array([f(x_i[0], x_i[1]) for x_i in x])

# Obtenemos el índice del 10% de los elementos aleatorio
tam = int(np.floor(len(y) * 0.1))
seleccionados = np.random.randint(0, N, tam)

# Cambiamos el signo de la etiqueta de los elementos seleccionados
y[seleccionados] *= -1

# Modificamos el vector de características para que sea de la forma (1, x1, x2)
x = np.insert(x, 0, 1, axis=1)

# Separamos las dos clases
clase_1 = np.array([x_i for x_i, y_i in zip(x, y) if y_i == -1])
clase_2 = np.array([x_i for x_i, y_i in zip(x, y) if y_i == 1])

# Pintamos el mapa de etiquetas obtenido
plt.scatter(clase_1[:, 1], clase_1[:, 2], c='purple')
plt.scatter(clase_2[:, 1], clase_2[:, 2], c='yellow')
plt.show()

# Ajustamos un modelo de regresión lineal a los datos mediante la pseudoinversa
print("Calculando SGD ...")
w,_,_ = sgd(x, y,
            lr=0.01,
            grad_fun=grad_Err,
            fun=Err,
            epsilon=10e-8,
            epochs=200,
            batch_size=64)

# Mostramos el error dentro de la muestra obtenido
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))

#%%
# Realizamos el experimento repitiendo lo mismo de arriba 1000 veces, sin
# pintar los gráficos
def experimento():
    N = 1000
    size = 1
    d = 2
    n = 100
    Ein = np.empty((n))
    Eout = np.empty((n))
    for i in range(n):
        print(i)
        # Generamos los datos de entrenamiento
        x = simula_unif(N, d, size)
        y = np.array([f(x_i[0], x_i[1]) for x_i in x])
        # Generamos otros datos para el test
        x_test = simula_unif(N, d, size)
        y_test = np.array([f(x_i[0], x_i[1]) for x_i in x])
        
        # Obtenemos el índice del 10% de los elementos aleatorio
        tam = int(np.floor(len(y) * 0.1))
        seleccionados = np.random.randint(0, N, tam)

        # Cambiamos el signo de la etiqueta de los elementos seleccionados
        y[seleccionados] *= -1
        y_test[seleccionados] *= -1

        # Modificamos el vector de características para que sea de la forma (1, x1, x2)
        x = np.insert(x, 0, 1, axis=1)
        x_test = np.insert(x_test, 0, 1, axis=1)

        w,_,_ = sgd(x, y,
                    lr=0.01,
                    grad_fun=grad_Err,
                    fun=Err,
                    epsilon=10e-8,
                    epochs=60,
                    batch_size=128)
        
        Ein[i] = Err(x, y, w)
        Eout[i] = Err(x_test, y_test, w)
        
    print("Ein medio: ", np.average(Ein))
    print("Eout medio: ", np.average(Eout))

experimento()

#%%
N = 1000
size = 1
d = 2
x = simula_unif(N, d, size)

# Obtenemos la etiqueta de cada elemento con la función f
y = np.array([f(x_i[0], x_i[1]) for x_i in x])

# Obtenemos el índice del 10% de los elementos aleatorio
tam = int(np.floor(len(y) * 0.1))
seleccionados = np.random.randint(0, N, tam)

# Cambiamos el signo de la etiqueta de los elementos seleccionados
y[seleccionados] *= -1

# Modificamos el vector de características para que sea de la forma
# (1, x1, x2, x1x2, x1^2, x2^2)
x = np.insert(x, 0, 1, axis=1)
aux = np.array([[x_i[1] * x_i[2], x_i[1]**2, x_i[2]**2] for x_i in x])
x = np.array([np.concatenate((x_i, aux_i)) for x_i, aux_i in zip (x, aux)])

print("Calculando SGD ...")
w,_,_ = sgd(x, y,
            lr=0.01,
            grad_fun=grad_Err,
            fun=Err,
            epsilon=10e-8,
            epochs=200,
            batch_size=64) 

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))