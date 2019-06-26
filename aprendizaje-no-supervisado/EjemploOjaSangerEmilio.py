
from OjaSanger import *
import matplotlib.pyplot as plt

ros = redOjaSanger();

n = 2;
L_in = [];
L_ref = [10,200,10,100];
largo = 300;
div = 100;

L_temp = [];
for i in L_ref:
        L_curr = np.random.normal(1,i,largo)
        s = sum(L_curr)
        m = s/len(L_curr)
        for j in range(len(L_curr)):
                L_curr[j] = L_curr[j]-m;
        L_temp.append(L_curr[:])

for i in range(largo):
        L_in.append([]);
        for j in range(len(L_temp)):
                L_in[i].append(L_temp[j][i]/div);

n_train = 0.04;
w_tolerance = 0.0001;
w_init = 100;
OjaSanger = 1; # 0 para Oja, 1 para Sanger

ros.OjaSanger(n,L_in,n_train=n_train,w_tolerance=w_tolerance,w_init=w_init,OjaSanger=OjaSanger);

if len(L_ref) == 2 and n == 1:
        #Creo un grafico con los valores relevantes

        #L_graph para parsear los datos de input
        L_graph = L_in[:];
        plt.scatter(*zip(*L_graph),marker='.',c='k');

        #L_vector para parsear los vectores ojasanger
        L_vector = []
        for i in ros.w_OjaSanger_log_[:-1]:
                L_vector.append((i[0][0],i[0][1]))

        plt.scatter(*zip(*L_vector),marker='.',c='b')
                
        plt.plot(ros.w_OjaSanger_log_[-1][0][0],ros.w_OjaSanger_log_[-1][0][1], marker='x',c='r')

        M_L_ref = max([max(i) for i in L_in])

        plt.xlabel('x');
        plt.ylabel('y');
        plt.xlim(-M_L_ref*1.1,M_L_ref*1.1);
        plt.ylim(-M_L_ref*1.1,M_L_ref*1.1);
        plt.show();
