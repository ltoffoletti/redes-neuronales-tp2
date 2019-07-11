# SOM.py
# Mapas autoorganizados con python
# La idea es que sirva para cualquier conjunto de datos

import numpy as np
import matplotlib.pyplot as plt
import random

class ClaseSOM(object):
    def __init__(self, filename, dim=3, rows=10, cols=10, epoch=10000, learn_rate=0.5):
        self.dim = dim
        self.rows = rows
        self.cols = cols
        self.learn_rate = learn_rate  # Tasa de aprendizaje inicial y máxima
        self.epoch = epoch

        self.range_max = self.rows + self.cols  # Distancia maxima de manhatan

        # Lectura de datos

        data_file = np.load(filename)
        docu_file = data_file[:, :self.dim]
        categoria = data_file[:, self.dim]

        for i in range(len(categoria)):
            categoria[i] = int(categoria[i])

        # Normalización en caso de ser necesario
        # docu_file = standarize_data(np.array(docu_file))

        #########################################################################################################

        # Seleccionar datos de entrenamiento

        docu_todo = list(zip(docu_file, categoria))
        np.random.shuffle(docu_todo)

        docu_file = np.array([docu_todo[i][0] for i in range(len(docu_todo))])
        categoria = np.array([docu_todo[i][1] for i in range(len(docu_todo))])

        n_train = 750

        self.docu_entrada = docu_file[:n_train]
        self.categoria = categoria[:n_train]

        self.docu_validacion = docu_file[n_train:]
        self.categoria_validacion = categoria[n_train:]
        ##########################################################################################################

    ##########################################################################################################
    # Algunas funciones útiles:




    # Devuelve los índices fila y columna en el mapa SOM que son las coordenadas de la celda en en el mapa
    # cuyo vector está más cerca del vector patron seleccionado en el conjunto de datos de netrada
    # El vector de celda mas cercano se le denomina generalmente Mejor Unidad de Coincidencia(MUC) en SOM
    def MUC(self, input_data, t, map, m_rows, m_cols):
        result = (0, 0)
        small_dist = 1.0e20
        for i in range(m_rows):
            for j in range(m_cols):
                ed = euc_dist(map[i][j], input_data[t])
                if ed < small_dist:
                    small_dist = ed
                    result = (i, j)
        return result


    ## Devuelve el valor más común o que más se repite de una lista de valores :
    def most_common(self, lst, n):
        # lst es la lista de valores 0 . . n
        if len(lst) == 0: return 0
        count = np.zeros(shape=n, dtype=np.int)
        for i in range(len(lst)):
            count[int(lst[i])] += 1
        return np.argmax(count)


    def standarize_data(self, input_data):
        X = input_data
        # print(np.mean(input_data, axis = 0))
        X -= np.mean(input_data, axis=0)
        X /= np.std(input_data, axis=0)
        return X


    #########################################################################################################
    def create_map(self):
        # SOM INIT

        # create SOM
        SOM_map = np.random.random_sample(size=(self.rows, self.cols, self.dim))  # Instancia inicial del mapa SOM

        for paso in range(self.epoch):
            if paso % (self.epoch / 10) == 0: print("step = ", str(paso))
            pct_left = 1.0 - ((paso * 1.0) / self.epoch)  # Porcenraje de iteracion restante
            curr_range = int(pct_left * self.range_max)  # maxiam distancia de manhatan en la que el "paso" decide "cerrar"
            # print (curr_range)
            learn_rate_update = pct_left * self.learn_rate
            # print (learn_rate_update)

            pattern = np.random.randint(len(self.docu_entrada))  # seleccionamos aleatoriamente un patron en datos de entrada
            # print (docu_entrada.shape,pattern,SOM_map.shape)
            (bmu_row, bmu_col) = self.MUC(self.docu_entrada, pattern, SOM_map, self.rows,
                                     self.cols)  # determina el nodo en unidad de mapa que mejor coincida

            # Examinamos cada nodo en SOM
            for i in range(self.rows):
                for j in range(self.cols):
                    # si dicho nodo está cerca de MUC: actualizamos el nodo del vector actual
                    if manhattan_dist(bmu_row, bmu_col, i,
                                      j) < curr_range:  # verificamos si la distancia de manhatan desde MUC es menor que max dist de manhatan
                        SOM_map[i][j] = SOM_map[i][j] + learn_rate_update * (self.docu_entrada[pattern] - SOM_map[i][j])

                        # else:
                    #    SOM_map[i][j] = SOM_map[i][j] - learn_rate_update * (docu_entrada[pattern] - SOM_map[i][j])

            # print("Se ha completado el mapa SOM \n")
            # La actualización acerca el vector del nodo actual al patron de datos utilizando
            # el valor de learn_rate_update que disminuye lentamente con el tiempo.

        # print (SOM_map.shape)

        # Reducción de dimensionalidad: Arreglos de dos dimensiones
        # Asociar cada nodo a la categoría a que pertenece

        print("Asociando cada nodo a su categoría")
        SOM_mapping2d = np.empty(shape=(self.rows, self.cols), dtype=object)  # mapa SOM de dos dimensiones
        for i in range(self.rows):
            for j in range(self.cols):
                SOM_mapping2d[i][j] = []
        for line in range(len(self.docu_entrada)):
            (m_row, m_col) = self.MUC(self.docu_entrada, line, SOM_map, self.rows, self.cols)
            SOM_mapping2d[m_row][m_col].append(self.categoria[line])

        # SOM_mapping2d sería nuestro mapa en dos dimensiones
        categoria_map = np.zeros(shape=(self.rows, self.cols), dtype=np.int)
        for i in range(self.rows):
            for j in range(self.cols):
                categoria_map[i][j] = self.most_common(SOM_mapping2d[i][j], 10)

        # Validacion
        print("Calculando validacion")
        SOM_mapping2d_validacion = np.empty(shape=(self.rows, self.cols), dtype=object)  # mapa SOM de dos dimensiones
        for i in range(self.rows):
            for j in range(self.cols):
                SOM_mapping2d_validacion[i][j] = []
        for line in range(len(self.docu_validacion)):
            (m_row, m_col) = self.MUC(self.docu_validacion, line, SOM_map, self.rows, self.cols)
            SOM_mapping2d_validacion[m_row][m_col].append(self.categoria_validacion[line])
        clasificaciones_ok = 0
        # SOM_mapping2d  de validacion
        categoria_map_validacion = np.zeros(shape=(self.rows, self.cols), dtype=np.int)
        for i in range(self.rows):
            for j in range(self.cols):
                categoria_map_validacion[i][j] = self.most_common(SOM_mapping2d_validacion[i][j], 10)
                for value in (SOM_mapping2d_validacion[i][j]):
                    if value == categoria_map[i][j]:
                        clasificaciones_ok += 1

        print("Clasificaciones OK: {0} sobre {1}, efectividad: {2}%".format(clasificaciones_ok,
                                                                           len(self.docu_validacion),
                                                                            round(clasificaciones_ok*100/len(self.docu_validacion),2)))

        # Imprime el mapa y la validacion

        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle('Self organized map', fontsize=16)
        cmap = plt.cm.get_cmap('rainbow', 10)
        cmap.set_under(color='white')
        axs[0].set_title('Entrenamiento')
        axs[0].imshow(categoria_map, cmap=cmap, vmin=0.001)

        axs[1].set_title('Validacion')
        axs[1].imshow(categoria_map_validacion, cmap=cmap, vmin=0.001)
        plt.show()


    # plt.imshow(map[:,:,1])
    # plt.colorbar()
    # plt.show()

# Distancia euclídea entre dos vectores
def euc_dist(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# Distancia de manhattan entre dos celdas con coordenadas (row1, col1) and (row2, col2)
def manhattan_dist(row1, col1, row2, col2):
    return np.abs(row1 - row2) + np.abs(col1 - col2)

# som = ClaseSOM()
# som.create_map()