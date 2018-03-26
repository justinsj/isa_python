#%%

###########################################################################################
###############         #################         ##########################    ###########
#################   ###################  #########  #####################  ###  ###########
#################  ##################  ################################  ####  ############
################  ####################  #############################  #####  #############
###############  ######################       #####################         ###############
##############  #############################  ###################  ######  ###############
#############  ####################  #########  ################  #######  ################
############  ####################  #########  ###############   #######  #################
########         ##### #############          ######## #####  #########  ####### ##########
###########################################################################################
###########################################################################################


#%%

#import libraries

#Imported library in Define_Node_Loc_2.0.3 or later
#import os
#from statistics import mean
#import numpy as np
#from shapely.geometry import Point, LineString, box
#from shapely.affinity import scale, rotate, translate
#from shapely.ops import cascaded_union
#import math

import os

from shapely.geometry import MultiLineString
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pylab as pl

#%%
def to_excel(matrix, name_of_file):
   import xlsxwriter
   file = str(name_of_file)+".xlsx"
   workbook = xlsxwriter.Workbook(file)
   worksheet = workbook.add_worksheet()
   row = 0
   matrix = matrix.tolist()
   for row in range(len(matrix)):
       for col in range(len(matrix[0])):
           worksheet.write(row, col, matrix[row][col])
   workbook.close()
   
#%%

#Graphical Output of data
def plot():
    global fig; global ax
    fig, ax = pl.subplots(figsize=(30, 20))
    plt.rc("font", size = 25)
    for i in range(len(element_list)):
        start = Point(node_list[element_list[i][1]][1:3])
        end   = Point(node_list[element_list[i][2]][1:3])
        ax.add_patch(Polygon(np.array(LineString((start, end)).buffer(0.005).exterior), color='gray'))
    ax.set_xlim(-1-.1*max(np.array(node_list)[:, 1:2]), 1+1.1*max(np.array(node_list)[:, 1:2]))
    ax.set_ylim(-1-.1*max(np.array(node_list)[:, 2:3]), 1+1.1*max(np.array(node_list)[:, 2:3]))  
    ax.set_aspect("equal")
    

def plot_displacement(scale):
    plot()
    for i in range(len(element_dis)):
        start = Point(scale*np.array(L_matrixs[i]*u_vector).transpose()[0, 0:2] + node_dis[element_dis[i][1]][1:3])
        end   = Point(scale*np.array(L_matrixs[i]*u_vector).transpose()[0, 3:5] + node_dis[element_dis[i][2]][1:3])
        ax.add_patch(Polygon(np.array(LineString((start, end)).buffer(0.01).exterior), color='black'))
    plt.title("Displacement Diagram"+"\n", fontweight="bold")
    plt.show()
    
def plot_bending_moment(scale):
    plot()
    for i in range(len(element_dis)):
        start_beam = np.array(node_dis[element_dis[i][1]][1:3])
        end_beam = np.array(node_dis[element_dis[i][2]][1:3])
        theta  = math.atan2((end_beam - start_beam)[1], (end_beam - start_beam)[0]) *180/math.pi
        moment = sum((mem_f_vectors[i])[2::3].tolist(), [])
        start_moment = rotate(LineString((Point(start_beam), Point(start_beam[0], start_beam[1]+scale*0.01*moment[0]))), theta, origin=Point(start_beam))
        end_moment =  rotate(LineString((Point(end_beam), Point(end_beam[0], end_beam[1]-scale*0.01*moment[1]))), theta, origin=Point(end_beam))
        connect = LineString((Point(start_moment.coords[-1]), Point(end_moment.coords[-1])))
        shape = MultiLineString((start_moment, connect, end_moment))
        ax.add_patch(Polygon(np.array(shape.buffer(0.01).exterior), color='red'))
    plt.title("Bending Moment Diagram"+"\n", fontweight="bold")
    plt.show()

def plot_shear_force(scale):
    plot()
    for i in range(len(element_dis)):
        start_beam = np.array(node_dis[element_dis[i][1]][1:3])
        end_beam = np.array(node_dis[element_dis[i][2]][1:3])
        theta  = math.atan2((end_beam - start_beam)[1], (end_beam - start_beam)[0]) *180/math.pi
        moment = sum((mem_f_vectors[i])[1::3].tolist(), [])
        start_moment = rotate(LineString((Point(start_beam), Point(start_beam[0], start_beam[1]+scale*0.01*moment[0]))), theta, origin=Point(start_beam))
        end_moment =  rotate(LineString((Point(end_beam), Point(end_beam[0], end_beam[1]-scale*0.01*moment[1]))), theta, origin=Point(end_beam))
        connect = LineString((Point(start_moment.coords[-1]), Point(end_moment.coords[-1])))
        shape = MultiLineString((start_moment, connect, end_moment))
        ax.add_patch(Polygon(np.array(shape.buffer(0.01).exterior), color='blue'))
    plt.title("Shear Force Diagram"+"\n", fontweight="bold")
    plt.show()   

def plot_axial_force(scale):
    plot()
    for i in range(len(element_dis)):
        start_beam = np.array(node_dis[element_dis[i][1]][1:3])
        end_beam = np.array(node_dis[element_dis[i][2]][1:3])
        theta  = math.atan2((end_beam - start_beam)[1], (end_beam - start_beam)[0]) *180/math.pi
        moment = sum((mem_f_vectors[i])[::3].tolist(), [])
        start_moment = rotate(LineString((Point(start_beam), Point(start_beam[0], start_beam[1]+scale*0.01*moment[0]))), theta, origin=Point(start_beam))
        end_moment =  rotate(LineString((Point(end_beam), Point(end_beam[0], end_beam[1]-scale*0.01*moment[1]))), theta, origin=Point(end_beam))
        connect = LineString((Point(start_moment.coords[-1]), Point(end_moment.coords[-1])))
        shape = MultiLineString((start_moment, connect, end_moment))
        ax.add_patch(Polygon(np.array(shape.buffer(0.01).exterior), color='green'))
    plt.title("Axial Force Diagram"+"\n", fontweight="bold")
    plt.show()  

#%%
file_des = "C:\\Users\\jimmy\\OneDrive - HKUST Connect\\Academics\\FYP\\Code\\"

node_list = []
element_list = []
node_dis = []
element_dis = []

K_matrixs = []
L_matrixs = []
R_matrixs = []
Fapp_vectors = []
mem_f_vectors = []

discretize = 25

#Define parameters

K_constant  = [[[  0,   0,   0,   0,   0,   0],
                [  0,  12,   6,   0, -12,   6],
                [  0,   6,   4,   0,  -6,   2],
                [  0,   0,   0,   0,   0,   0],
                [  0, -12,  -6,   0,  12,  -6],
                [  0,   6,   2,   0,  -6,   4]],


               [[  0,   0,   0,   0,   0,   0],
                [  0,   3,   2,   0,   3,   2],
                [  0,   2,   1,   0,   2,   1],
                [  0,   0,   0,   0,   0,   0],
                [  0,   3,   2,   0,   3,   2],
                [  0,   2,   1,   0,   2,   1]]]
              
A = 0.02
E = 200000000
I = 0.0002

#%%
#Import ground truth

no = input("Which set of ground truth should be used? "+"\n")

check = True
trial = int(no)
code = "Define_Node_Loc_2.0.3.py"

exec(open(os.path.join(file_des, code)).read(), globals())

del check

#%%
"""
#Discretization of element
GCD = math.hypot((node_list[0][1] - node_list[1][1]), (node_list[0][2] - node_list[1][2]))
for i, j,  in zip(node_list[1::], node_list[2::]):
    GCD = fractions.gcd(math.hypot((i[1] - j[1]), (i[2] - j[2])), GCD)

total_L = sum([math.hypot(node_list[i[1]][1]-node_list[i[2]][1], node_list[i[1]][2]-node_list[i[2]][2]) for i in element_list])
discretize = total_L/int(math.ceil(discretize*GCD/total_L)*total_L/GCD)

for i in element_list:
    x1 = node_list[i[1]][1]; y1 = node_list[i[1]][2]; x = x1; y = y1
    x2 = node_list[i[2]][1]; y2 = node_list[i[2]][2]
    l = math.hypot((x2 - x1), (y2 - y1))
    add = 0
    while (x, y) != (x2, y2):
        add += 1
        x = x1 + add * discretize/l * (x2 - x1); y = y1 + add * discretize/l * (y2 - y1)
        node_dis.append([  "x",    x,    y,    0,    0,    0,    0,    0,    0])
    node_dis.pop(-1)

node_dis.extend(node_list)
node_dis = sorted(node_dis, key = lambda x: (x[1], x[2]))


for i in element_list:
    n1 = [index for index, x in enumerate(node_dis) if x[0] == i[1]][0]
    n2 = [index for index, x in enumerate(node_dis) if x[0] == i[2]][0]
    for j in range(n1, n2):
        element_dis.append([j, j+1, i[3]])
        
node_dis = [[index]+i[1::] for index, i in enumerate(node_dis)]
element_dis = [[index]+i for index, i in enumerate(element_dis)]
"""
"""
#Discretization of element
temp = list(set(filter(None, sum([[abs(node_list[i[2]][1] - node_list[i[1]][1]), abs(node_list[i[2]][2] - node_list[i[1]][2])] for i in element_list], []))))
GCD = temp[0]
for i in temp[1::]:
    GCD = fractions.gcd(i, GCD)
discretize = np.prod(temp)/GCD**(len(temp)-1)
del temp[:]
"""

for index, i in enumerate(element_list):
    x1 = node_list[i[1]][1]; y1 = node_list[i[1]][2]; x = x1; y = y1
    x2 = node_list[i[2]][1]; y2 = node_list[i[2]][2]
    add = 0
    while (x, y) != (x2, y2):
        add += 1
        x = x1 + add * (x2 - x1)/discretize; y = y1 + add * (y2 - y1)/discretize
        node_dis.append([  "x",    x,    y,    0,    0,    0,    0,    0,    0])
    node_dis.pop(-1)
    
node_dis.extend(node_list)
node_dis = sorted(node_dis, key = lambda x: (x[1], x[2]))
    
for i in element_list:
    n1 = np.array([x[1:3] for x in node_dis if x[0] == i[1]])[0]
    n2 = np.array([x[1:3] for x in node_dis if x[0] == i[2]])[0]
    d = round(math.hypot((n2-n1)[0], (n2-n1)[1]), 5)
    for index,j in enumerate(node_dis):
        n = np.array(j[1:3])
        d1 = round(math.hypot((n-n1)[0], (n-n1)[1]), 6)
        d2 = round(math.hypot((n-n2)[0], (n-n2)[1]), 6)
        if round(d1 + d2, 5) == d:
            temp.append(index)
    for x, y in zip(temp[::], temp[1::]):
        element_dis.append([x, y , i[3]])
    del temp[:]
   
node_dis = [[index]+i[1::] for index, i in enumerate(node_dis)]
element_dis = [[index]+i for index, i in enumerate(element_dis)]


#%%

#K_matrix(s) & Assembly

for i in element_dis:
    L = math.hypot((node_dis[i[1]][1] - node_dis[i[2]][1]), (node_dis[i[1]][2] - node_dis[i[2]][2]))
    K_mat = [[  A*E/L,      0,     0,-A*E/L,     0,     0],
             [      0,      0,     0,     0,     0,     0],
             [      0,      0,     0,     0,     0,     0],
             [ -A*E/L,      0,     0, A*E/L,     0,     0],
             [      0,      0,     0,     0,     0,     0],
             [      0,      0,     0,     0,     0,     0]]
    for x in (1,2,4,5):
        for y in (1,2,4,5):
            K_mat[x][y] = K_constant[0][x][y]*E*I/(L**K_constant[1][x][y])
    K_matrixs.append(np.matrix(K_mat))
    del K_mat
    
    L_mat = np.zeros((6,3*len(node_dis)))
    for index, j in enumerate([[0,3],[3,6]]):
        L_mat[j[0]:j[1], 3*i[index+1]:3*i[index+1]+3] = np.identity(3)
    L_matrixs.append(np.matrix(L_mat))
    del L_mat
    
    r = math.atan2((node_dis[i[2]][2] - node_dis[i[1]][2]) , node_dis[i[2]][1] - node_dis[i[1]][1])
    R_mat = [[ math.cos(r), math.sin(r),           0,           0,           0,           0],
             [-math.sin(r), math.cos(r),           0,           0,           0,           0],
             [           0,           0,           1,           0,           0,           0],
             [           0,           0,           0, math.cos(r), math.sin(r),           0 ],
             [           0,           0,           0,-math.sin(r), math.cos(r),           0],
             [           0,           0,           0,           0,           0,           1]]
    R_matrixs.append(np.matrix(R_mat))
    del R_mat
    
    Fapp_vec = [[0], [0], [0], [0], [0], [0]]
    if i[3] != 0:
        #F = wL/2       M = wL^2/12
        w = i[3] * L / 2
        m = w * L  / 6
        
        Fapp_vec[1][0] += w
        Fapp_vec[4][0] += w
        Fapp_vec[2][0] += m
        Fapp_vec[5][0] -= m

    Fapp_vectors.append(np.matrix(Fapp_vec))
    del Fapp_vec

#%%
#Assembly of Stiffness matrix
K_matrix = sum([k.transpose()*j.transpose()*i*j*k for i, j, k in zip(K_matrixs, R_matrixs, L_matrixs)])

temp = [bool(y)^1 for y in sum([x[6:9:] for x in node_dis], [])]
Bcon = np.matrix(np.diag(temp)[~np.all(np.diag(temp) == 0, axis=0)])
del temp[:]

Fapp_vector = sum([j.transpose()*i for i, j in zip(Fapp_vectors, L_matrixs)]) + np.matrix(sum([x[3:6:] for x in node_dis], [])).transpose()

u_vector = Bcon.transpose()*np.linalg.inv(Bcon*K_matrix*Bcon.transpose())*Bcon*Fapp_vector
Frec_vector = K_matrix*u_vector - Fapp_vector

temp = [bool(y)^0 for y in sum([x[6:9:] for x in node_dis], [])]
Bcon_reverse = np.matrix(np.diag(temp)[~np.all(np.diag(temp) == 0, axis=0)])

print("\n",["%.2f" % x[0] for x in Bcon_reverse*Frec_vector.tolist()])

for i in range(len(element_dis)):
    mem_f_vectors.append(K_matrixs[i]*(R_matrixs[i]*L_matrixs[i]*u_vector) - Fapp_vectors[i])

#%%
plot_displacement(250)
plot_axial_force(3)
plot_shear_force(3)
plot_bending_moment(3)
