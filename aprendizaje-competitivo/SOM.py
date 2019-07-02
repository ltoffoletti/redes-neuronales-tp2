# SOM.py
# Mapas autoorganizados con python
# La idea es que sirva para cualquier conjunto de datos

import numpy as np
import matplotlib.pyplot as plt
import random


##########################################################################################################
# Algunas funciones útiles:

# Distancia euclídea entre dos vectores
def euc_dist(vec1, vec2):
  return np.linalg.norm(vec1 - vec2) 
 
# Distancia de manhattan entre dos celdas con coordenadas (row1, col1) and (row2, col2)
def manhattan_dist(row1, col1, row2, col2):
  return np.abs(row1-row2) + np.abs(col1-col2)


# Devuelve los índices fila y columna en el mapa SOM que son las coordenadas de la celda en en el mapa 
# cuyo vector está más cerca del vector patron seleccionado en el conjunto de datos de netrada
# El vector de celda mas cercano se le denomina generalmente Mejor Unidad de Coincidencia(MUC) en SOM 
def MUC(input_data, t, map, m_rows, m_cols):
  result = (0,0)
  small_dist = 1.0e20
  for i in range(m_rows):
    for j in range(m_cols):
      ed = euc_dist(map[i][j], input_data[t])
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result


## Devuelve el valor más común o que más se repite de una lista de valores :
def most_common(lst, n):
  # lst es la lista de valores 0 . . n
  if len(lst) == 0: return -1
  count = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    count[lst[i]] += 1
  return np.argmax(count)

def standarize_data(input_data):
    X = input_data
    #print(np.mean(input_data, axis = 0))
    X -= np.mean(input_data, axis = 0)
    X /= np.std(input_data, axis = 0)
    return X


#########################################################################################################
def main():  
    np.random.seed(1)
    dim = 850    
    rows = 20; cols = 20
    range_max = rows + cols   # Distancia maxima de manhatan
    learn_rate = 0.5          # Tasa de aprendizaje inicial y máxima
    epoch = 7000

    # Lectura de datos

    data_file = 'tp2_training_dataset.csv'
    docu_file = np.loadtxt(data_file, delimiter=",", usecols=range(1,851),
        dtype=np.float)
    categoria = np.loadtxt(data_file, delimiter=",", usecols=[0],
        dtype=np.int)
    

    
    # Normalización en caso de ser necesario
    #docu_entrada = standarize_data(np.array(docu_entrada))
    
    #########################################################################################################
    
    # Seleccionar datos de entrenamiento 
    porciento_train = 0.8
    train = random.choice(list(range(len(docu_file))))     
    train_set_size = int(train* porciento_train)
    print (train,train_set_size,len(docu_file))

    docu_entrada = docu_file[0:train_set_size] 


    ##########################################################################################################

    # SOM INIT

    # create SOM
    SOM_map = np.random.random_sample(size=(rows,cols,dim))                    # Instancia inicial del mapa SOM 
    
    for paso in range(epoch):
        if paso % (epoch/10) == 0: print("step = ", str(paso))
        pct_left = 1.0 - ((paso * 1.0) / epoch)                                # Porcenraje de iteracion restante
        curr_range = (int)(pct_left * range_max)                               # maxiam distancia de manhatan en la que el "paso" decide "cerrar"
        #print (curr_range) 
        learn_rate_update = pct_left * learn_rate
        #print (learn_rate_update)
        
        
        pattern = np.random.randint(len(docu_entrada))                          # seleccionamos aleatoriamente un patron en datos de entrada
        #print (docu_entrada.shape,pattern,SOM_map.shape)
        (bmu_row, bmu_col) = MUC(docu_entrada, pattern, SOM_map, rows, cols)    #determina el nodo en unidad de mapa que mejor coincida 
        

        # Examinamos cada nodo en SOM 
        for i in range(rows):
            for j in range(cols):
                # si dicho nodo está cerca de MUC: actualizamos el nodo del vector actual  
                if manhattan_dist(bmu_row, bmu_col, i, j)  < curr_range:    # verificamos si la distancia de manhatan desde MUC es menor que max dist de manhatan
                    SOM_map[i][j] = SOM_map[i][j] + learn_rate_update * (docu_entrada[pattern] - SOM_map[i][j]) 
                
                #else:
                #    SOM_map[i][j] = SOM_map[i][j] - learn_rate_update * (docu_entrada[pattern] - SOM_map[i][j]) 
         
        #print("Se ha completado el mapa SOM \n") 
        # La actualización acerca el vector del nodo actual al patron de datos utilizando 
        # el valor de learn_rate_update que disminuye lentamente con el tiempo.
        
    #print (SOM_map.shape)
     
    # Reducción de dimensionalidad: Arreglos de dos dimensiones 
    # Asociar cada nodo a la categoría a que pertenece

    print("Asociando cada nodo a su categoría")
    SOM_mapping2d = np.empty(shape=(rows,cols), dtype=object)                # mapa SOM de dos dimensiones
    for i in range(rows):
        for j in range(cols):
            SOM_mapping2d[i][j] = []
    for line in range(len(docu_entrada)):
        (m_row, m_col) = MUC(docu_entrada, line, SOM_map, rows, cols)
        SOM_mapping2d[m_row][m_col].append(categoria[line])
    
    # SOM_mapping2d sería nuestro mapa en dos dimensiones    
    categoria_map = np.zeros(shape=(rows,cols), dtype=np.int)
    for i in range(rows):
        for j in range(cols):
            categoria_map[i][j] = most_common(SOM_mapping2d[i][j], 10)
    
    #plt.imshow(SOM_map[:,:,0], cmap=plt.cm.get_cmap('terrain_r'))                                     
    plt.imshow(categoria_map, cmap=plt.cm.get_cmap('terrain_r'))
    plt.colorbar()
    plt.show()   

"""               
######################################################################################################################################################################                 
    ### modelo equivalente que utilice como entrada los datos proyectados en las primeras 3 componentes principales 

    dim = 9
    rows = 30; cols = 30
    range_max = rows + cols   # Distancia maxima de manhatan
    learn_rate = 0.5          # Tasa de aprendizaje inicial y máxima
    epoch = 5000

    # SOM INIT 

    # create SOM
    SOM_map = SOM_map[:,:,0:dim].reshape(30*30,dim)
    SOM_map_component = np.random.random_sample(size=(rows,cols,dim))                    # Instancia inicial del mapa SOM 
    
    for paso in range(epoch):
        if paso % (epoch/10) == 0: print("step = ", str(paso))
        pct_left = 1.0 - ((paso * 1.0) / epoch)                                # Porcenraje de iteracion restante
        curr_range = (int)(pct_left * range_max)                               # maxiam distancia de manhatan en la que el "paso" decide "cerrar"
        #print (curr_range) 
        learn_rate_update = pct_left * learn_rate
        #print (learn_rate_update)
        
        #print (SOM_map.shape)
        

        pattern = np.random.randint(len(SOM_map))                                    # seleccionamos aleatoriamente un patron en datos de entrada
        #print (pattern,len(docu_entrada))
        (bmu_row, bmu_col) = MUC(SOM_map, pattern, SOM_map_component, rows, cols)    #determina el nodo en unidad de mapa que mejor coincida 
        

        # Examinamos cada nodo en SOM 
        for i in range(rows):
            for j in range(cols):
                # si dicho nodo está cerca de MUC: actualizamos el nodo del vector actual  
                if manhattan_dist(bmu_row, bmu_col, i, j)  < curr_range:    # verificamos si la distancia de manhatan desde MUC es menor que max dist de manhatan
                    SOM_map_component[i][j] = SOM_map_component[i][j] + learn_rate_update * (SOM_map[pattern] - SOM_map_component[i][j]) 
                
                #else:
                #    SOM_map[i][j] = SOM_map[i][j] - learn_rate_update * (docu_entrada[pattern] - SOM_map[i][j]) 
         
        #print("Se ha completado el mapa SOM \n") 
        # La actualización acerca el vector del nodo actual al patron de datos utilizando 
        # el valor de learn_rate_update que disminuye lentamente con el tiempo.
        
    #print (SOM_map.shape)
     
    # Reducción de dimensionalidad: Arreglos de dos dimensiones 
    # Asociar cada nodo a la categoría a que pertenece

    print("Asociando cada nodo a su categoría")
    SOM_mapping2d = np.empty(shape=(rows,cols), dtype=object)                # mapa SOM de dos dimensiones
    for i in range(rows):
        for j in range(cols):
            SOM_mapping2d[i][j] = []
    for line in range(len(docu_entrada)):
        (m_row, m_col) = MUC(SOM_map, line, SOM_map_component, rows, cols)
        SOM_mapping2d[m_row][m_col].append(categoria[line])
    
    # SOM_mapping2d sería nuestro mapa en dos dimensiones    
    categoria_map = np.zeros(shape=(rows,cols), dtype=np.int)
    for i in range(rows):
        for j in range(cols):
            categoria_map[i][j] = most_common(SOM_mapping2d[i][j], 10)
                                         
    plt.imshow(categoria_map, cmap=plt.cm.get_cmap('terrain_r'))
    plt.colorbar()
    plt.show() 

"""                                     
      
if __name__=="__main__":
  main()   


# plt.imshow(map[:,:,1])
# plt.colorbar()
# plt.show()

