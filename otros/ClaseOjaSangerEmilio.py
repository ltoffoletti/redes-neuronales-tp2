import math
import numpy as np
import random
import os

class redOjaSanger(object):
        '''
        Red Neuronal para calculo de componentes principales
        '''
        #Inicializa la matriz de pesos con valores pequeños
        def __init__(self,Dim=[2,2,1],w_init=10,beta=1,verbose=7):
                #Dim es una lista con la dimension de cada capa
                #Dim[-1]==output // Dim[0]==input
                #w_init es un maximo/minimo para los pesos iniciales 
                #(Se generan entre w_init/100 y -w_init/100)
                self.w_ = [];
                for i in range(len(Dim)-1):
                        #Llena self.w_ con un array por espacio entre capas (len(Dim)-1 capas)
                        #Cada elemento de self.w_ es un array de numpy con todos los pesos de esa capa (iniciados al azar)
                        self.w_.append(np.array([[random.randint(-w_init,w_init)/100 for i in range(Dim[i]+1)] for j in range(Dim[i+1])]));

                self.Dim = Dim;
                self.beta = beta;
                #Esto es alto para testing
                self.verbose = verbose;

                print('\nLa red neuronal fue creada con exito.')
                if self.verbose >= 10:
                        print('Matriz de pesos (incluye bias):\n')
                        for x in self.w_:
                                print(x)
                print('')

        #Derivada de la funcion identidad
        def _derivada_identidad(self,n,b):
                return b

        #Funcion identidad
        def _funcion_identidad(self,n,b):
                return n*b

        #Funcion que hace la sumatoria de w(ki)*y(k) para todos los valores de output
        def _suma_Oja(self,output,y):
                cont = 0;
                #Se recorre cada posicion del output
                #z es k en la teorica
                for z in range(len(output)):
                        cont += self.w_OjaSanger[z,y]*output[z];
                return cont

        #Funcion para actualizar los pesos segun la regla de Oja
        def _Oja_update(self,n_train,current,output):
                #Se recorre cada fila de la matriz de pesos (j en teorica)
                for x in range(len(self.w_OjaSanger)):
                        #Se recorre cada entrada de cada fila de la matriz de pesos (cada columna, i en teorica)
                        for y in range(len(self.w_OjaSanger[x])):
                                self.w_OjaSanger[x,y] += n_train*output[x]*(current[y]-self._suma_Oja(output,y))
                return self

        #Funcion para actualizar los pesos segun la regla de Sanger
        def _Sanger_update(self,n_train,current,output):
                #Se recorre cada fila de la matriz de pesos (j en teorica)
                for x in range(len(self.w_OjaSanger)):
                        #Se recorre cada entrada de cada fila de la matriz de pesos (cada columna, i en teorica)
                        for y in range(len(self.w_OjaSanger[x])):
                                self.w_OjaSanger[x,y] += n_train*output[x]*(current[y]-self._suma_Sanger(output,y,x)-self.w_OjaSanger[x,y]*output[x])
                return self

        #Funcion que hace la sumatoria de w(ki)*y(k) para los valores de output menores a j-1 (j en teorica)
        def _suma_Sanger(self,output,y,x):
                cont = 0;
                #Se recorre cada valor del output hasta la neurona x-1 (j-1 en teorica)
                #z es k en la teorica
                for z in range(x):
                        cont += self.w_OjaSanger[z,y]*output[z];
                return cont

        #Funcion para revisar si el cambio de pesos entre epocas es significativo
        def _w_OjaSanger_check(self,tolerance):
                change = 0;
                #Se recorre cada fila de la matriz de pesos
                for x in range(len(self.w_OjaSanger_log_[-1])):
                        #Se recorre cada entrada de cada fila de la matriz de pesos (cada columna)
                        for y in range(len(self.w_OjaSanger_log_[-1][x])):
                                change += (self.w_OjaSanger_log_[-1][x][y]-self.w_OjaSanger_log_[-2][x][y])**2
                if math.sqrt(change) <= tolerance:
                        print('El valor de cambio fue menor al tolerado. Se termina el entrenamiento.\nCambio del vector:')
                        print(math.sqrt(change))
                return math.sqrt(change) > tolerance

        #Funcion que devuelve n componentes principales segun regla de Oja
        def OjaSanger(self,n,L_in,n_train=0.05,w_tolerance=0.001,w_init=50,OjaSanger=0):
                #n es la cantidad de componentes principales que devuelve la funcion
                #L_in es la lista que contiene los datos (solo input)
                #L_in debe estar centrado en 0

                #Primero inicializo los pesos de los componentes principales
                self.w_OjaSanger = np.array([[random.randint(-w_init,w_init)/100 for x in range(len(L_in[0]))] for y in range(n)]);

                print('\nLa red neuronal para Oja-Sanger fue creada con exito.')
                if self.verbose >= 7:
                        print('Matriz de pesos (no incluye bias):')
                        for x in self.w_OjaSanger:
                                print(x)
                print('')

                #Inicializo el log de pesos
                self.w_OjaSanger_log_ = [np.copy(self.w_OjaSanger)];

                #i cuenta un minimo de iteraciones por si se aplana la funcion al principio
                i=0;
                #b revisa que el cambio de pesos sea significativo
                b=True;
                while b:

                        #Coeficiente de correccion del n_train
                        corr_n = n_train*w_tolerance*i

                        #Una vez cada vuelta por L_in se reordena la lista
                        if i%len(L_in) == 0:
                                random.shuffle(L_in);
                                if self.verbose >= 7 and i > 0:
                                        print('Iteracion numero ' + str(i));

                        #current representa el input actual
                        current = L_in[i%len(L_in)];

                        #output es el resultado de input*pesos
                        output = np.dot(self.w_OjaSanger,current);

                        #Despues actualizo el peso
                        if OjaSanger == 0:
                                #Actualizacion segun regla de Oja
                                self._Oja_update(n_train,current,output);
                        elif OjaSanger == 1:
                                #Actualizacion segun regla de Sanger
                                self._Sanger_update(n_train-corr_n,current,output);
                        else:
                                print('Mensaje de error 1/2 por variable OjaSanger.')

                        if self.verbose >= 8:
                                print('Matriz de pesos actualizada:\n')
                                for k in self.w_OjaSanger:
                                        print(k)
                                print('')

                        #Se agrega el peso actualizado al log
                        self.w_OjaSanger_log_.append(np.copy(self.w_OjaSanger));

                        #Reviso si el cambio de pesos es significativo
                        b = i < int(math.sqrt(len(L_in))) or self._w_OjaSanger_check(w_tolerance);

                        i+=1;

                if OjaSanger == 0:
                        print('\nSe ha finalizado el entrenamiento segun la regla de Oja.\nPesos finales:');
                elif OjaSanger == 1:
                        print('\nSe ha finalizado el entrenamiento segun la regla de Sanger.\nPesos finales:');
                else:
                        print('Mensaje de error 2/2 por variable OjaSanger.')

                for x in self.w_OjaSanger:
                        print(x);

                return self.w_OjaSanger
