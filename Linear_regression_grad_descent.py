# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:54:23 2017

@author: vc185059
"""
import numpy
#import matplotlib.pyplot as plt

with open('input2.csv') as f:
    trainingsamples = f.readlines()
    
ages=[]
weights=[]
heights=[]
for i in trainingsamples:
    fs=i.split('\n')
    features=fs[0].split(',')
    ages.append(float(features[0]))
    weights.append(float(features[1]))
    heights.append(float(features[2]))

#meanage=numpy.mean(ages)
#meanweight=numpy.mean(weights)
#meanheight=numpy.mean(heights)
meanage=sum(ages)/len(ages)
meanweight=sum(weights)/len(weights)
meanheight=sum(heights)/len(heights)
stdage=numpy.std(ages)
stdweight=numpy.std(weights)
stdheight=numpy.std(heights)

for i in range(len(ages)):
    ages[i]=(ages[i]-meanage)/stdage
    weights[i]=(weights[i]-meanweight)/stdweight
    #heights[i]=(heights[i]-meanheight)/stdheight
#print(ages)
#print(weights)
f=open('output2.csv','w')
w=[0,0,0]
lrs=[0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,1]
for lr in lrs:
    w=[0,0,0]
    for itera in range(100):
        cost=0
        gradin=0
        gradientage=0
        gradientweight=0
        for i in range(len(ages)):
            out=w[0]+w[1]*ages[i]+w[2]*weights[i]
            #cost+=(out-heights[i])**2
            gradin+=(out-heights[i])
            gradientage+=(out-heights[i])*ages[i]
            gradientweight+=(out-heights[i])*weights[i]
        #cost/=(2*len(ages))
        gradientage/=len(ages)
        gradientweight/=len(ages)
        gradin/=len(ages)
        
        w[0]-=lr*gradin
        w[1]-=lr*gradientage
        w[2]-=lr*gradientweight

    #plt.figure()
    #plt.scatter(ages,weights,heights,'r')
    #plt.show()
    output=[]
    for i in range(len(ages)):
        output.append(w[0]+w[1]*ages[i]+w[2]*weights[i])
    #plt.figure()
    #plt.scatter(ages,weights,output,'b')
    cost=0
    for i in range(len(ages)):
        out=w[0]+w[1]*ages[i]+w[2]*weights[i]
        cost+=(out-heights[i])**2
    print(lr,cost)
    print('\n')        
    print(lr,100,w[0],w[1],w[2])
    f.write(str(lr)+','+str(100)+','+str(w[0])+','+str(w[1])+','+str(w[2]))
    f.write('\n')
    print('\n')    
f.close()