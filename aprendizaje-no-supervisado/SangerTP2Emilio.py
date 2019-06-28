
from OjaSanger import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

###################################################################################

#Nombre del archivo con los sets de entrenamiento
nombre = 'tp2_training_dataset.csv';

#Numero de componentes principales a extraer
n = 3;

#Variables de entrenamiento
n_train = 0.01;
w_tolerance = 0.0001;
w_init = 100;
OjaSanger = 1; # 0 para Oja, 1 para Sanger

#Abro el archivo
with open(os.path.dirname(os.path.abspath(__file__)) + '/' + str(nombre),'r') as F:
    L = F.readlines();
    L_out = [];
    L_in = [];
    for i in range(len(L)):
        L[i] = L[i].rstrip('\n').split(',');
        L_out.append(L[i][0]);
        L_in.append(L[i][1:]);

#Transformo los datos de input en numeros
for i in range(len(L_in)):
    for j in range(len(L_in[i])):
        L_in[i][j] = int(L_in[i][j]);

#Inicializo la red y la entreno
ros = redOjaSanger(verbose=7);
W = ros.OjaSanger(n,L_in,n_train=n_train,w_tolerance=w_tolerance,
                  w_init=w_init,OjaSanger=OjaSanger);

#Lista de colores para el grafico
clist = [int(i)/9 for i in L_out];

###Lista cmaps### (https://matplotlib.org/gallery/color/colormap_reference.html)
#Diverging:
#'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
#'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'

#Perceptually Uniform Sequential:
#'viridis', 'plasma', 'inferno', 'magma', 'cividis'

#Qualitative:
#'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1',
#'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'

cmap = 'tab10'; #Todavia no se cual queda mejor

#Hago una lista con los resultados de la red para todos los datos de L_in
salida = []
for i in L_in:
    salida.append(np.dot(W,i));

#Paso los resultados a tres listas de dimensiones
xdata = [];
ydata = [];
zdata = [];
for i in salida:
    xdata.append(i[0]);
    ydata.append(i[1]);
    zdata.append(i[2]);

ax = plt.axes(projection='3d')
ax.scatter3D(xdata, ydata, zdata, c=clist, cmap=cmap);
plt.show()
