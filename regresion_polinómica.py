import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Crear el DataSet"""

x = np.arange(0,2*np.pi, 0.1)
y = [np.sin(e+np.random.normal(0, 0.2)) for e in x]
plt.plot(x, y, "*")

import random
entrenamiento = random.sample((x), len(x)*0.7)
print(type(x))
print(type(entrenamiento))

from pandas.core.indexes.multi import names_compat
size = len(entrenamiento.index)
k = len(entrenamiento.columns)-1

print(size, k)

x = np.array([[i*1.0 for i in range(k)] for j in range(size)])
y = np.array([])

for i in range(size):
  lista = []
  for j in range(k):
    lista.append(entrenamiento_x.iloc[i][entrenamiento_x.columns[j]])
  lista = np.array(lista)
  x[i] = lista
  y = np.append(y, entrenamiento_y.iloc[i])

mat = [[i*1.0 for i in range(size)] for j in range(k)]
for i in range(k):
  col = x[:,i]
  max_x = max(col)
  min_x = min(col)
  lista = []
  for e in col:
    lista.append( (e-min_x) / (max_x - min_x) )
  lista = np.array(lista)
  mat[i] = lista

x_norm = np.vstack((mat)).T

max_y = max(y)
min_y = min(y)
y_norm = np.array([ ( e - min_y)/(max_y - min_y) for e in y])

print(x_norm.size)
print(y_norm.size)

print(x_norm)
print(y_norm)

"""# Modelo 
$h(x_i) = x_i*w + b$

# Sección nueva
"""

def h(x,w):
  return np.dot(x, w.transpose())

"""$Error = \frac{1}{2m}\sum_{i=0}^m (y_i - h(x_i)) ²$

"""

def Error(x,y,w):
  e = 0
  for i in range(size):
    e = e + (y[i]-h(x[i], w))**2/(2.0*size)
  return e

"""$db = \frac{1}{m}\sum_{i=0}^m(y_i - h(x_i))(-1)$

$dw = \frac{1}{m}\sum_{i=0}^m(y_i - h(x_i))(-x_i)$ 
"""

def derivada(x,y,w):
  #implementar las derivadas
  db = sum([(y[i] - h(x[i],w))*-1 for i in range(size)])/(size*1.0)
  dw = [db]
  for j in range(1, k):
    dw.append(sum([(y[i] - h(x[i], w))*-x[i][j] for i in range(size)])/(size*1.0))
  return dw

def update(w, alfa, dw):
  for i in range(w.size):
    w[i] = w[i] - alfa*dw[i]
  return w

def train(x,y,umbral, alfa):
  w = np.array([np.random.rand() for i in range(k)])

  x[0] = 1
  L = Error(x,y,w)
  errorAcumulado = []

  while(L > umbral):
    dw = derivada(x,y,w)
    w = update(w,alfa,dw) 
    errorAcumulado.append(L)
    # print(L)
    L = Error(x,y,w) 

  return errorAcumulado, w

def test(x,y,w):
  #y_pred = h(x,w,b)
  plt.plot(x,y,'*')
  plt.plot(x,[h(xi,w) for xi in x])
  #return y_pred - y  
  # graficar puntos y la y aproximado.

errorAcumulado, w = train(x_norm, y_norm, 0.05, 0.01)
plt.plot([i for i in range(len(errorAcumulado))], errorAcumulado, '*')

def main(umbral, alfa):
   errorAcumulado, w = train(x_norm, y_norm, 0.05, 0.01)
   size = x.size
   x_test = [i*0.7 for i in range(size)]
   y_pred = [test(x_test[i], h) for i in range(size)]

main(0.05, 0.01)