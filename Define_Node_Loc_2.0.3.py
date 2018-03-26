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

#Define_Node_loc

#2.0.0
#Welcome to 2D world

#%%

#import libraries
import os
from statistics import mean
import numpy as np
from shapely.geometry import Point, LineString, box
from shapely.affinity import scale, rotate, translate
from shapely.ops import cascaded_union
import math

#%%

#Define name of groups
temp = []

items = []

number = []
force_moment = []
noise = []
support = []
UDL = []
beam = []
length = []
height = []
measure_length = []
measure_height = []

beam_line = []
beam_ends = []

node = []
mid_node = []
element = []

coord_x = []
coord_y = []
node_x = []
node_y = []

node_table = []
element_table = []

class_type = {tuple(range(10)): "number",
              tuple(range(10, 22)): "force_moment",
              (22, 23): "noise",
              tuple(range(24, 48)): "support",
              (48,): "UDL",
              tuple(range(52,56)): "beam",
              (56,): "length",
              (57,): "height",
              tuple(range(60, 62)): "measure_length",
              tuple(range(62, 64)): "measure_height"
              }

loading_table = {(10,):            ["+", 3],
                 (11,):            ["-", 3],
                 (12,):            ["+", 2],
                 (13,):            ["-", 2],
                 (18, 19, 20, 21): ["+", 4],
                 (14, 15, 16, 17): ["-", 4] }
    
support_table = {tuple(range(24,32)):  [5,6,7],
                 tuple(range(32,40)):  [5,6]  ,
                 (40, 41, 44, 45):     [6]    ,
                 (42, 43, 46, 47):     [7]     }


class_name = [  '0','1','2','3','4','5','6','7','8','9',
                #10
                'upwards force','downwards force','rightwards force','leftwards force',
                #14
                'counter-clockwise moment right', 'counter-clockwise moment up', 'counter-clockwise moment left', 'counter-clockwise moment down', 
                #18
                'clockwise moment right','clockwise moment up','clockwise moment left','clockwise moment down',
                #22
                'unknown','random alphabet',
                #24
                'fixed support right','fixed support left','fixed support down', 'fixed support up',
                #28
                'fixed support right w/ beam','fixed support left w/ beam','fixed support down w/ beam', 'fixed support up w/ beam',
                #32
                'pinned support down', 'pinned support up','pinned support left', 'pinned support right',
                #36
                'pinned support down w/ beam', 'pinned support up w/ beam','pinned support left w/ beam', 'pinned support right w/ beam',
                #40
                'roller support down', 'roller support up','roller support left','roller support right',
                #44
                'roller support down w/ beam', 'roller support up w/ beam','roller support left w/ beam','roller support right w/ beam',
                #48
                'uniformly distributed load', 'linearly distributed load','quadratically distributed load', 'cubically distributed load',
                #52
                'horizontal beam','vertical beam','downward diagonal beam', 'upward diagonal beam',
                #56
                'length','height','counter-clockwise angle','clockwise angle',
                #60
                'measure left','measure right','measure up','measure down',
                #64
                'beam' #for display purpose
                ]

#Values/ Strings that are to changed

#File Input - Destination
file_des = "C:\\Users\\jimmy\\OneDrive - HKUST Connect\\Academics\\FYP\\Code\\"

#Threshold value
duplicate_node = 60
coord_width = 30
relevent_digit = 50
mid_range = 0.3
num_range = 0.4

#For checking purpose

#%%

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pylab as pl
import matplotlib.image as mpimg

def show(name):
    global color ; global color_ref
    fig, ax = pl.subplots(figsize=(20,13.5))
    color_ref = {tuple(range(10)): 0,
                  tuple(sum((tuple(range(10, 22)),(48,)), ())): 1,
                  tuple(range(24, 48)): 2,
                  (64,): 3,
                  (56, 60, 61, 62): 4,
                  (57, 62, 63, 64): 5
                  
                  }
    color_no = ['#ff00ff', '#ff0000', '#0000ff', '#828282', "#40ff00", "#40ff00"]
    #color = ["#ff4000","#ff8000","#ffbf00","#ffff00","#bfff00","#80ff00","#40ff00","#00ff00","#00ff40","#00ff80","#00ffbf","#00ffff","#00bfff","#0080ff","#0040ff","#4000ff","#8000ff","#bf00ff","#ff00ff","#ff00bf","#ff0080","#ff0040","#ff0000"]
    for i in name:
        color = color_no[next(y for x, y in color_ref.items() if i[1] in x)]
        shape = Polygon(np.array(i[0].exterior), alpha=0.7, fc=color)
        ax.add_patch(shape)
        ax.annotate(class_name[i[1]], i[0].centroid.coords[0], color='black', weight='bold', 
                fontsize=20, ha='center', va='center')
    ax.set_xlim(0, 1150)
    ax.set_ylim(0, 780)
    plt.show()

#%%

def beam2line(beam):
    global beam_end; global beam_linestring
    bound = beam[0].bounds
    func_ref = {52: [Point(bound[0], mean([bound[1], bound[3]])), Point(bound[2], mean([bound[1], bound[3]]))],
                53: [Point(mean([bound[0], bound[2]]), bound[1]), Point(mean([bound[0], bound[2]]), bound[3])],
                54: [Point(bound[0], bound[3]), Point(bound[2], bound[1])],
                55: [Point(bound[0], bound[1]), Point(bound[2], bound[3])] }
    beam_end = func_ref[beam[1]]
    beam_linestring = LineString([beam_end[0], beam_end[1]])
    
def ellipse(mid_pt, semi_a_v, theta):
    c = Point(mid_pt).buffer(1)
    ell  = scale(c, int(semi_a_v[0]), int(semi_a_v[1]))
    ellr = rotate(ell,theta)
    return ellr
    
def line2ellipse(line):
    a = Point(line.coords[0]); b = Point(line.coords[1])
    mid_pt = ((a.x + b.x)*0.5 ,  (a.y + b.y)*0.5)
    #semi_a_v = (math.hypot(a.x - b.x, a.y - b.y)*mid_range[1], math.hypot(a.x - b.x, a.y - b.y)*mid_range[0])
    semi_a_v = (math.hypot(a.x - b.x, a.y - b.y)*mid_range, duplicate_node)
    theta = round(180*math.atan((a.y-b.y)/ (a.x-b.x+0.1))/math.pi, 5)
    return ellipse(mid_pt, semi_a_v, theta)
    
#%%

#Import ground truth
try:
    if check:
        no = str(trial)
        plt_show = False
    else:
        plt_show = True
        no = input("Which set of ground truth should be used? "+"\n")
except:
    plt_show = True
    no = input("Which set of ground truth should be used? "+"\n")

GT = "GT_all_"+str(no)+".txt"
file = os.path.join(file_des + "GT", (GT))
for i in open(file).readlines()[1:]:
    item = i.rstrip().split("\t")
    item = list(map(int, item[0].split()))
    items.append(item)
items = sorted(items, key = lambda x: (x[0]+ 0.25*x[1]))
    
#Sort objects into different list
for i in items:
    shape = box(i[0], 780-i[1]-i[3], i[0]+i[2], 780-i[1])
    group = next(y for x, y in class_type.items() if i[4] in x)
    globals()[group].append([shape, i[4]])
    #Transform beam object to linestring and end 
    if group == "beam":
        beam2line([shape,i[4]])
        beam_line.append(beam_linestring)
        beam_ends.extend(beam_end)

for i, j, k, l in zip((measure_length, measure_height), ([60,61], [62,63]),(length, height), (56, 57)):
    for x, y in zip(i[::], i[1::]):
        if x[-1] == j[0] and y[-1] == j[1]:
            a = cascaded_union([x[0], y[0]]).bounds
            k.append([box(a[0], a[1], a[2], a[3]), l])
            
#%%

if plt_show == True:
    #"""
    plt.figure(figsize = (21.2,14.3))
    img=mpimg.imread(os.path.join(file_des + "Images", ("all_image_done_"+no+".png")))
    imgplot = plt.imshow(img)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    plt.show()
    #"""
    show(number+force_moment+UDL+support+[[i.buffer(5), 64] for i in beam_line]+length+height)
   
#%%

#Combine Number objects into actual numbers
for index, i in enumerate(number):
    for j in number[index+1::]:
        distance = i[0].distance(j[0])
        if distance <= relevent_digit:
            digit = int(str(i[1])+ str(j[1]))
            a = cascaded_union([i[0], j[0]]).bounds
            i[:] = [box(a[0], a[1], a[2], a[3]), digit]
            number.remove(j)
            
#Assign combined number to length and force/moment/UDL    
#Assign number to force, moment, UDL
for i in (force_moment + UDL + support + length + height):
    
    #Identify the beam that the loading is acting on, name = corr_beam
    if i not in (length + height + UDL): 
        for j in beam_line:
            temp.append(i[0].distance(j))
        corr_beam = beam_line[temp.index(min(temp))]
        del temp[:]
    
    if i in (length + height + UDL):
        for j in beam_line:
            pt = j.interpolate(j.project(i[0].centroid))
            length_moved = translate(i[0], pt.x-i[0].centroid.x, pt.y-i[0].centroid.y).buffer(10)
            trim_beam = length_moved.intersection(j)
            if trim_beam.within(length_moved):
                temp.append(i[0].distance(length_moved))
            else:
                temp.append(99999)
        corr_beam = beam_line[temp.index(min(temp))]
        del temp[:]

    #Only number that is on the same side of the beam is of interest
    if i not in support:
        for j in number:
            if LineString([j[0].centroid, i[0].centroid]).intersects(corr_beam):
                temp.append(999999)
            else:
                temp.append(i[0].distance(j[0]))
        relevent_num = number[temp.index(min(temp))]
        i.append(relevent_num[1])
        number.remove(relevent_num)
        del temp[:]
    #Add node in case UDL or measurement object is at the middle of beam
    #if i in (UDL + length + height):
    """
    if i in (UDL):
        box = i[0].bounds
        if box[2] - box[0] > box[3] - box[1]:
            node.append(corr_beam.interpolate(corr_beam.project(Point(box[0], mean([box[1], box[3]])))))
            node.append(corr_beam.interpolate(corr_beam.project(Point(box[2], mean([box[1], box[3]])))))
        else:
            node.append(corr_beam.interpolate(corr_beam.project(Point(mean([box[0], box[2]]), box[1]))))
            node.append(corr_beam.interpolate(corr_beam.project(Point(mean([box[0], box[2]]), box[3]))))
    """
    #Project object onto corresponding beam
    if i in support:
        c = i[0].centroid
        new_c = corr_beam.interpolate(corr_beam.project(c))
        node.append(new_c)
    del corr_beam   
#%%
#Create Coordination System 
if bool(length+height):
    for i, j in zip((length, height), (coord_x, coord_y)):
        for k in i:
            bound = k[0].bounds
            if i == length:
                if j == []:
                    j.append([LineString([Point(bound[0], 0), Point(bound[0], 780)]).buffer(coord_width), 0])
                j.append([LineString([Point(bound[2], 0), Point(bound[2], 780)]).buffer(coord_width), j[-1][-1]+k[-1]])
            else:
                if j == []:
                    j.append([LineString([Point(0, bound[1]), Point(1150, bound[1])]).buffer(coord_width), 0])
                j.append([LineString([Point(0, bound[3]), Point(1150, bound[3])]).buffer(coord_width), j[-1][-1]+k[-1]])
                    
for i in [x[0] for x in coord_x+coord_y]:
    for j in beam_line:
        if i.intersects(j) and i.distance(Point(j.coords[0])) >= duplicate_node and i.distance(Point(j.coords[1])) >= duplicate_node:
            node.append(i.intersection(j).centroid)
    
#%%
#Create node by the ends of node and the supoort 

node = list(sorted(node + beam_ends, key=lambda x: x.x + x.y))
node = [i.buffer(duplicate_node) for i in node]
for index, i in enumerate(node):
    for j in node[index+1::]:
        if i.intersects(j):
            node[index] = Point((i.centroid.x + j.centroid.x)*0.5 , (i.centroid.y + j.centroid.y)*0.5).buffer(duplicate_node)
            node.remove(j)
    
#%%
#When measurement object is absent
if not bool(length)^bool(height):
    num_range = (1-num_range)/2
    for a in node:
        pt = a.centroid
        for i, j in zip((node_x, node_y), (pt.x, pt.y)):
            i.append(j)
            for x in i[:-1:]:
                if abs(j - x) <= duplicate_node:
                    i.remove(j)
                    break

    for i, j, k in zip((node_x, node_y), (coord_x, coord_y), (0,1)):
        for index, x in enumerate(i):
            for y in i[index+1::]:
                for z in number:
                    if int(z[0].centroid.coords[0][k]) in range(int((1-num_range)*x + num_range*y), int(num_range*x + (1-num_range)*y)):
                        if j == coord_x:
                            if j == []:
                                j.append([LineString([Point(x, 0), Point(x, 780)]).buffer(coord_width), 0])
                            j.append([LineString([Point(y, 0), Point(y, 780)]).buffer(coord_width), j[-1][-1]+z[-1]])
                        else:  
                            if j == []:
                                j.append([LineString([Point(0, x), Point(1150, x)]).buffer(coord_width), 0])
                            j.append([LineString([Point(0, y), Point(1150, y)]).buffer(coord_width), j[-1][-1]+z[-1]])
                        number.remove(z)
                        break
                continue

#%%
#Split beam object to form element        
for index, i in enumerate(beam_line):
    for j in node:
        if j.distance(i) < duplicate_node and (j.distance(Point(i.coords[0])) > duplicate_node and j.distance(Point(i.coords[1])) > duplicate_node):
            element.append(LineString([Point(i.coords[0]), j.centroid]))
            i = (LineString([j.centroid, Point(i.coords[1])]))
    element.append(i)
    
for index, i in enumerate(element):
    for a in (0, 1):
        for b in range(len(node)):
            if Point(i.coords[a]).intersects(node[b]):
                temp.append(b)
                break
    element[index] = [i, temp[0], temp[1]]
    del temp[:]
    mid_node.append(line2ellipse(i))

#%%
truth = 1
trial_count = 0

while truth != []:
    mid_node = []
    node_table = []
    element_table = []
    truth = list(force_moment + support + UDL)
    duplicate_node += trial_count*10
    trial_count += 1
    node = [x[0].centroid.buffer(duplicate_node) if type(x) == list else x.centroid.buffer(duplicate_node) for x in node]
    mid_node = [line2ellipse(x[0]) for x in element]
    
    #Create node table
    for index, i in enumerate(node):
        node[index] = [i, 0, 0]
        node_table.append([0, 0, 0, 0, 0, 0, 0, 0])
        for x, y in zip((coord_x, coord_y), (0, 1)):
            for z in x:
                if i.intersects(z[0]):
                    node_table[-1][y] = z[-1]
                    node[index][y+1] = z[-1]
                    break
        for j in force_moment:
            if i.intersects(j[0]):
                truth.remove(j)
                place = next(y for x, y in loading_table.items() if j[1] in x)
                node_table[-1][place[1]] = int(place[0]+str(j[-1]))
        for j in support:
            if i.intersects(j[0]):
                truth.remove(j)
                place = next(y for x, y in support_table.items() if j[1] in x)
                for l in place:
                    node_table[-1][l] = 1
    
    for i, j in zip(element, mid_node):
        for x in force_moment:
            if j.intersects(x[0]):
                truth.remove(x)
                node_table.append([0, 0, 0, 0, 0, 0, 0, 0])
                place = next(b for a, b in loading_table.items() if x[1] in a)
                node_table[-1][place[1]] = int(place[0]+str(x[-1]))
                
                node_table[-1][0] = mean([node[i[1]][1], node[i[2]][1]])
                node_table[-1][1] = mean([node[i[1]][2], node[i[2]][2]])
                del temp[:]
                
    node_table = list(sorted(node_table, key=lambda x:int(str(int(x[1]))+str(int(x[0])))))
    node_table = np.hstack((np.array([[i] for i in range(len(node_table))]), np.array(node_table)))           
    
    #Create Element Table
    for i, j in zip(element, mid_node):
        element_table.append([0,0,0,0,0])
        for a in (1,2):
            element_table[-1][2*a-2:2*a:] = node[i[a]][1::]
            
        for x in UDL:
            if x[0].intersects(j) and x[0].intersects(node[i[1]][0]) and x[0].intersects(node[i[2]][0]):
                truth.remove(x)
                element_table[-1][4] = int("-"+str(x[-1]))

    for index, i in enumerate(element_table):
        line = LineString([Point(i[0:2:]), Point(i[2:4:])])
        for j in [Point(x[1:3]) for x in node_table]:
            if line.intersects(j) and line.project(j) != 0:
                temp.append([i[0], i[1], j.x, j.y, i[4]])
                i[:] = [j.x, j.y, i[2], i[3], i[4]]
        element_table[index] = list(temp)
        del temp[:]
    
    element_table = list(sum(element_table, []))
    element_list = []
    
    for i in element_table:
        temp = [0,0,i[-1]]
        for j, k in zip((0,1), (i[:2:], i[2:4])):
            for index, l in enumerate(node_table):
                if k == l[1:3:].tolist():
                    temp[j] = index
                    break
        element_list.append(list(temp))
        del temp[:]

    element_table = list(sorted(element_list, key=lambda x:str(x[0])+str(x[1])))
    element_table = np.hstack((np.array([[i] for i in range(len(element_list))]), np.array(element_list)))           
    
#%%
#Plot the processed image 
if plt_show == True:
    fig, ax = pl.subplots(figsize=(20, 13.5))
    
    for i in [i[0] for i in force_moment] + [i[0] for i in UDL] :
        shape = Polygon(np.array(i.exterior), alpha=0.7, fc='#ff0000')
        ax.add_patch(shape)
        
    for i in [i[0] for i in support]:
        shape = Polygon(np.array(i.exterior), alpha=0.7, fc='#0000ff')
        ax.add_patch(shape)
        
    for i in [i[0].buffer(5) for i in element]:
        shape = Polygon(np.array(i.exterior), alpha=0.7, fc='#000000')
        ax.add_patch(shape)
    
    for i in [i[0] for i in node]+mid_node:
        shape = Polygon(np.array(i.exterior), alpha=0.3, fc='#ff00ff')
        ax.add_patch(shape)
        
    for i in [i[0] for i in coord_x + coord_y]:
        shape = Polygon(np.array(i.exterior), alpha=0.3, fc='#00ff33')
        ax.add_patch(shape)
    
    #ax.autoscale()
    #ax.relim()
    ax.set_xlim(0, 1150)
    ax.set_ylim(0, 780)
    
    plt.show()

print("\n"+"Node Table: "+"\n")
print(node_table)
print("\n"+"Element Table: "+"\n")
print(element_table)

node_list = node_table.tolist()
element_list= element_table.tolist()