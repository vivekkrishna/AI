# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 14:47:55 2017

@author: vc185059
"""
#import pylab as plt
#import matplotlib.pyplot as plt
with open('input1.csv') as f:
    trainingsamples = f.readlines()
    
w=[0,0,0]
xvalues1=[]
yvalues1=[]
xvalues2=[]
yvalues2=[]
isconverged=False
f=open('output1.csv','w')
while(not isconverged):
    prevw=w[:]
    for i in trainingsamples:
        a=i.split('\n')
        b=a[0].split(',')
        if int(b[2])==1:
            xvalues1.append(b[0])
            yvalues1.append(b[1])
        else:
            xvalues2.append(b[0])
            yvalues2.append(b[1])
        #print(b)
        y=int(b[0])*w[0]+int(b[1])*w[1]+1*w[2]
        out=0
        if y>0:
            out=1
        else:
            out=-1
        if out*int(b[2])<=0:
            w[0]=w[0]+int(b[2])*int(b[0])
            w[1]=w[1]+int(b[2])*int(b[1])
            w[2]=w[2]+int(b[2])
    if prevw==w:
        isconverged=True
    else:
        print(w)
        f.write(str(w[0])+','+str(w[1])+','+str(w[2])+'\n')
    if isconverged:
        break
         
f.close()
#plt.plot(xvalues1,yvalues1,'bo')
#plt.plot(xvalues2,yvalues2,'ro')
xvalues1=[]
yvalues1=[]
for i in [0,15]:
    j=-(i*w[0]+1*w[2])/w[1]
    xvalues1.append(i)
    yvalues1.append(j)
#plt.plot(xvalues1,yvalues1,'g-')
        

    