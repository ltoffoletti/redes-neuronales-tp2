from ClaseOjaSangerEmilio import *
import matplotlib.pyplot as plt
import os
import numpy as np
from mpl_toolkits.mplot3d import axes3d
###################################################################################
class RunnerTP2(object):
    def __init__(self, l_rate=0.01, w_tolerance=0.0001, w_init=100, OjaSanger=1, principal_components=3):
        # Numero de componentes principales a extraer
        self.pcs = principal_components;

        # Variables de entrenamiento
        self.l_rate = l_rate
        self.w_tolerance = w_tolerance
        self.w_init = w_init
        self.OjaSanger = OjaSanger  # 0 para Oja, 1 para Sanger
        self.weights = []
        self.salida = []

    # Entrena y guarda los datos
    def train(self, filename='data/tp2_training_dataset.csv'):
        # Abro el archivo
        with open(os.path.dirname(os.path.abspath(__file__)) + '/' + str(filename), 'r') as F:
            L = F.readlines()
            L_out = []
            L_in = []
            for i in range(len(L)):
                L[i] = L[i].rstrip('\n').split(',')
                L_out.append(L[i][0])
                L_in.append(L[i][1:])

        # Transformo los datos de input en numeros
        for i in range(len(L_in)):
            for j in range(len(L_in[i])):
                L_in[i][j] = int(L_in[i][j])
        self.L_out = L_out
        # Inicializo la red y la entreno
        ros = redOjaSanger(verbose=7)
        W = ros.OjaSanger(self.pcs, L_in, n_train=self.l_rate, w_tolerance=self.w_tolerance,
                          w_init=self.w_init, OjaSanger=self.OjaSanger)
        self.weights = W

        # Hago una lista con los resultados de la red para todos los datos de L_in
        salida = []
        for i in L_in:
            salida.append(np.dot(W, i))
        self.salida = salida
        self.save_results()

    def plot(self):
        ###Lista cmaps### (https://matplotlib.org/gallery/color/colormap_reference.html)
        # Diverging:
        # 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
        # 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'

        # Perceptually Uniform Sequential:
        # 'viridis', 'plasma', 'inferno', 'magma', 'cividis'

        # Qualitative:
        # 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
        # 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
        cmap = 'tab10'  # Todavia no se cual queda mejor

        # Lista de colores para el grafico
        clist = [int(i) / 9 for i in self.L_out]

        # Paso los resultados a tres listas de dimensiones
        xdata = []
        ydata = []
        zdata = []
        for i in self.salida:
            xdata.append(i[0])
            ydata.append(i[1])
            zdata.append(i[2])

        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata, ydata, zdata, c=clist, cmap=cmap)
        plt.show()

    # TODO si el nombre del archivo coincide se pisan los datos, se podria preguntar si se quiere pisar o guardar con otro valor
    def save_results(self):
        filename = "redes_tp2/" + \
                   "w_tolerance" + str(self.w_tolerance) + \
                   "l_rate" + str(self.l_rate) + \
                   "OjaSanger" + str(self.OjaSanger) + \
                   "w_init" + str(self.w_init) + \
                   "pcs" + str(self.pcs) + \
                   ".npz"
        np.savez(filename, w_tolerance=self.w_tolerance, l_rate=self.l_rate,
                 OjaSanger=self.OjaSanger, w_init=self.w_init,
                 salida=self.salida, weights=self.weights, pcs=self.pcs, L_out=self.L_out)

    def load_network(self, filename):
        data = np.load(filename)
        self.w_tolerance = data['w_tolerance'] or 0.0001
        self.l_rate = data['l_rate'] or 0.01
        self.OjaSanger = data['OjaSanger'] or 1
        self.w_init = data['w_init'] or 100
        self.pcs = data['pcs'] or 3
        self.weights = data['weights']
        self.salida = data['salida']
        self.L_out = data['L_out']
        #TODO agregar L_in??

    def datos_red(self):
        print("W_tolerance: {0}, learning_rate: {1} \n"
              "PCS: {2} \n"
              "W_init: {3} \n".format(self.w_tolerance, self.l_rate, self.pcs, self.w_init))
        if self.OjaSanger == 1:
            print("Regla de Sanger")
        else:
            print("Regla de Oja")

# Auxiliares
import glob

def opcion_entrenar():
    learning_rate = 0
    while learning_rate <= 0:
        learning_rate = float(input("Ingrese learning rate: "))
    print("Learning rate: {0}".format(learning_rate))
    w_tolerance = 0
    while w_tolerance <= 0:
        w_tolerance = float(input("Ingrese tolerancia de error: "))
    print("Tolerance: {0}".format(w_tolerance))
    w_init = 0
    while w_init <= 0:
        w_init = int(input("Ingrese w_init: "))
    print("W_init: {0}".format(w_init))
    OjaSanger = -1
    while OjaSanger != 0 and OjaSanger != 1:
        OjaSanger = int(input("Elija metodo para PCA (1-Sanger, 0-Oja: "))
    print("OjaSanger: {0}".format(OjaSanger))
    pcs = 0
    while pcs <= 0:
        pcs = int(input("Ingrese Nro de componentes principales a extraer: "))
    print("pcs: {0}".format(pcs))

    net = RunnerTP2(learning_rate, w_tolerance, w_init, OjaSanger, pcs)
    net.train()
    return net

def opcion_cargar():
    archivos = (glob.glob("redes_tp2/*.npz"))
    if len(archivos) == 0:
        print("No hay archivos para cargar.")
    else:
        for index, file in enumerate(archivos):
            print("{0} - {1}".format(index, file))
        nro_archivo = -1
        while nro_archivo <= -1 or nro_archivo > len(archivos):
            nro_archivo = int(input("Elija nro de archivo 0 a {0}: ".format(len(archivos) - 1)))

        net = RunnerTP2()
        net.load_network(archivos[nro_archivo])
        print("Cargado: {0}".format(archivos[nro_archivo]))
        net.plot()
    return net

# MAIN

salir = 0
while not salir:
    opcion = 0
    while opcion <= 0 or opcion > 5:
        opcion = int(input("\n 1-Entrenar\n 2-Cargar red desde archivo + Plot\n 3-Plot + "
                           "datos red actual \n 4-Entrenar con opciones por default \n 5-Salir \n Seleccione opcion: "))
    if opcion == 1:
        net = opcion_entrenar()
    elif opcion == 2:
        net = opcion_cargar()
    elif opcion == 3:
        if ('net' in locals()):
            net.datos_red()
            net.plot()
        else:
            print("No hay red cargada")
    elif opcion == 4:
        net = RunnerTP2()
        net.train()
    else:
        salir = 1