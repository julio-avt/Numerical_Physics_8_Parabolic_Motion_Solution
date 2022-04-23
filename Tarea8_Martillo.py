# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 02:16:01 2021

@author: Julio C. Torreblanca
"""

from pylab import *
import numpy as np
from scipy.integrate import odeint

# Definición de constantes
N = 1000         #No. de pasos
phi = np.pi/4   #Ángulo de lanzamiento
x0 = 0          #Posicion inicial en x
y0 = 2          #Posicion inicial en y
C = 0.       #Coeficiente de rozamiento
m = 7.26        #Masa del martillo
rho = 1.2       #Densidd del aire
R = 0.06        #Radio de la esfera
g = 9.81        #ac. gravedad
k = rho*np.pi*R**2*C/(m*2.)  #Constante global

#Semillas y criterio
t = 1.0        #Tiempo inicial
v0 = 10.       #Semilla inicial para la vel inicial 


#Función con la inicialización de las condiciones iniciales
def cond_ini(v0):
    """
    Esta función establece las condiciones iniciales para el arreglo dada 
    una velocidad inicial.
    
    Parámetros:
        v0: es la magnitud de la velocidad inicial
    Salida:
        array : [posición en x, vel en x, posición en y, vel en y]
    """
    m0 = np.float64(x0)           #Establecer estado inicial en x
    m1 = np.float64(v0*np.cos(phi)) #Vel. inicial en x
    m2 = np.float64(y0)             #Establecer estado inicial en y
    m3 = np.float64(v0*np.sin(phi))
    return array([m0,m1,m2,m3])

#Función que calculara a las f^(i) en la forma estándar
def f(y,t):
    """ 
    Esta función evalúa a las funciones f^(i) de la forma estandar para un 
    y en un tiempo t.
    
    Parámetros:
        y: arreglo [posicion en x, vel en x, posición en y, vel en y]
        t: arreglo con los puntos t_i a ser evaluados
    Salida:
        array : las funciones f^(i) evalada en los parámetors
                [f^(0), f^(1), f^(2), f^(3)]
    """
    f0 = y[1]
    f1 = -k*y[1]*(y[1]*y[1] + y[3]*y[3])**(1/2)
    f2 = y[3]
    f3 = -k*y[3]*(y[1]*y[1] + y[3]*y[3])**(1/2) - g
    return array([f0,f1,f2,f3])

def time(t,N):
    """
    Esta función calcula la solución para una vel. inicial v_0 dada a un tiempo
    t, y a través de un ciclo while aumenta el tiempo hasta generar la solución
    en la que el último punto de posición de y sea cero (choque con el suelo).
    
    Parámetros:
        t: tiempo en el que se generará la solución de 0 a t.
        N: no. de pasos en los que se dividirá el tiempo [0,t] para generar la 
           solución.
        
    Salida:
        respuesta: arreglo que contiene la solución numérica. En la primer 
                   columna se encuentras los valores para x, en la segunda 
                   columna la velocidad en x, en la tercera el valor de y, 
                   en la cuarta la velocidad en y. Todos estos valores son en
                   cada punto del tiempo en el arreglo llamado tiempo.
        tiempo: arreglo que contiene los puntos del tiempo donde se generó la
                solución numérica.

    """
    i=1      #Contador de iteraciones
    dy = 1.0 #Delta de y
    while dy > 0.3:
        #print(f'Iteracion y -> {i}')
        if i== 10000:       #Clausula para establecer un alto en caso de que calcule de más
            print('No. maximo de iteraciones alcanzado')
            break
        y = cond_ini(v0)                #Iniciamos el arreglo con las condiciones iniciales
        tiempo = linspace(0, t, N)      #Generamos los puntos para el tiemoo
        respuesta = odeint(f,y,tiempo)  #Generamos la solución numérica
        #print(respuesta)
        
        dy = abs(respuesta[-1][2] - 0.) #Verifica la condición para y_final
        
        t += 0.01   #Aumentamos el tiempo
        i += 1      #Aumentamos el contador
        
    return respuesta, tiempo



dx = 1.
j=1
while dx > 0.2:   
    if j== 10000:       #Clausula para establecer un alto en caso de que calcule de más
        print('No. maximo de iteraciones alcanzado')
        break
    res_noFric, tiempo_noFric = time(t,N)
    dx = abs(res_noFric[-1][0] - 86.745) #Verifica la confición del x final
    v0 += 0.1  #Aumentamos la velocidad inicial
    t = 1.     #Reseteamos el tiempo a 1 seg
    #print(f'x -> {j}')
    j+=1


print('-'*50)
print('Caso sin fricción:')
print(f' \tVel. inicial = {v0} m/s')
print(f'\tx final = {res_noFric[-1][0]} m')
print(f'\ty final = {res_noFric[-1][2]} m')
print(f'\tTiempo de vuelo = {tiempo_noFric[-1]} s')


# Ahora graficamos la solución para el caso sin fricción
figure('Ejercicio 1',figsize=(6,4))
plot(tiempo_noFric,res_noFric[:,2])
title('Gráfico de y(t) vs tiempo sin frición')
xlabel('$t$ [s]')
ylabel('$y(t)$ [m]')
grid()



####################### Cálculo para el flujo laminar
C = 0.5     #Flujo laminar
k = rho*np.pi*R**2*C/(m*2.)  #Constante global
t = 1.0        #Reseteamos el tiempo

res_laminar, tiempo_laminar = time(t,N)
    
print('-'*50)
print('Caso con flujo laminar:')
print(f' \tVel. inicial = {v0} m/s')
print(f'\tx final = {res_laminar[-1][0]} m')
print(f'\ty final = {res_laminar[-1][2]} m')
print(f'\tTiempo de vuelo = {tiempo_laminar[-1]} s')

####################### Cálculo para el flujo laminar
C = 0.75     #Flujo laminar
k = rho*np.pi*R**2*C/(m*2.)  #Constante global
t = 1.0        #Reseteamos el tiempo

res_ines, tiempo_ines = time(t,N)
    
print('-'*50)
print('Caso con flujo laminar:')
print(f' \tVel. inicial = {v0} m/s')
print(f'\tx final = {res_ines[-1][0]} m')
print(f'\ty final = {res_ines[-1][2]} m')
print(f'\tTiempo de vuelo = {tiempo_ines[-1]} s')


# Ahora graficamos las soluciones usando subgráficas
figure('Ejercicio 2',figsize=(8,7))
subplot(3,1,1)    # Definimos la primera gráfica
plot(tiempo_noFric,res_noFric[:,2], 'b',label='Sin fricción')
plot(tiempo_laminar,res_laminar[:,2], 'g',label='Flujo laminar')
plot(tiempo_ines,res_ines[:,2], 'r',label='Flujo inestable oscilante')
title('Gráfico de y(t) vs t para los tres casos')
xlabel('$t$ [s]')
ylabel('$y(t)$ [m]')
grid(True)
legend()    

subplot(3,1,3)    # Definimos la segunda gráfica
plot(res_noFric[:,0],res_noFric[:,2], 'b',label='Sin fricción')
plot(res_laminar[:,0],res_laminar[:,2], 'g',label='Flujo laminar')
plot(res_ines[:,0],res_ines[:,2], 'r',label='Flujo inestable oscilante')
title('Gráfico de y(t) vs x(t) para los tres casos')
xlabel('$x(t)$ [m]')
ylabel('$y(t)$ [m]')
grid(True)  
legend()   

 

##################Obtención de las diferencia de longitudes
dif_laminar = res_noFric[-1][0] - res_laminar[-1][0]
dif_ines = res_noFric[-1][0] - res_ines[-1][0]
print('-'*50)
print(f'Diferencia para el flujo laminar: {dif_laminar} m')
print(f'Diferencia para el flujo inestable oscilante: {dif_ines} m')
show()