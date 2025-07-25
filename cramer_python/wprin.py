import numpy
import sys
import textwrap
###from array import *
###from numpy import *

#
#
#

def main():
    prini(13,1)

###    a = numpy.array(range(1,26)).reshape((5, 5))

    mm=10000
    a = numpy.empty(mm,dtype='float')
    ka = numpy.empty(mm,dtype='int')

    for i in range(0,mm):
        a[i] = i*1.21- mm/2
        ka[i] = i- int(mm/2)


    b=[]
    b.append(1.0)
    b.append(2.0)
    print(b)

    b2=numpy.array(b)
    print(b2)

    c=numpy.ones([5,3])

    ijk=1
    for i in range(0,5):
        for j in range(0,3):
            c[i,j]=ijk
            ijk=ijk+1

###    c[2,2]=77777777
###    c[0,0]=1
###    c[0,1]=6
###    c[1,1]=12

    d=numpy.ones([5,3],'d')
    e=numpy.ones([5,3])
    len=6

    c[0] = 1 + 1e-14
    prin2('c, to start=',c,4)


    b[0]=1

    dif = b[0]-c[0]
    prin2('dif=',dif,1)

#
#    print giant arrays without truncation
#
    prin2('a=',a,mm)    
    prinf('ka=',ka,mm)    



###    exit()

###    prin2_long('b =',b)
    prin2_long('c =',c,len)
    prin2_long('d =',d,len)


    prin2('d =',d,len)

    print(c)
    prin2('c, again =',c,len)


    prinr2('c, as array=',c,2,3)

    prini(13,0)
    prin2('c again =',c,len)


    prini(0,0)
    prin2('c, not seen =',c,len)
    prin2_long('c, not seen =',c,len)


    prini(13,0)
    prin2('c, seen =',c,len)
    prin2_long('c, seen =',c,len)

#
#    print complex numbers
#
    zar = numpy.zeros([2,3],dtype=complex)

    zar[0,0]=.00000123456 + 1.23456j
    prinz('zar=',zar,len)


    print(zar)

    n=100
    ks = numpy.zeros(n,'int')
    ks[0]=1
    ks[2]=51000
    ks[3]=11000
    ks[n-1]=11

    prinf('ks=',ks,n)

###    print(ks)
###    print(zar)

#
#   printing arrays of integers
#
    m=5
    n=6
    kar = numpy.zeros([m,n],'int')

    for i in range(0,m):
        for j in range(0,n):
            kar[i,j] = i+j**2

    prinf('kar=',kar,12)
    print(kar)


    prin2('n=',n,1)
    prinf('n=',n,1)


    prinrf('karr=',kar,m,n)



    exit()

#
#
#

def prinrf(msg,a,m,n):
#
#    prints out first m-by-n submatrix
#
    if (prini.numf==0):
        return

    numpy.set_printoptions(
        formatter={'int': formatf},
        linewidth=prini.lw)

#
#    copy to smaller array
#
    a2=numpy.zeros([m,n],'int')
    for i in range(0,m):
        for j in range(0,n):
            a2[i,j]=a[i,j]
    a2=numpy.array([a2],'int')

    s=sys.stdout
    f = open(prini.name,'a')

    print_main(f,msg,a2)
    print_main(s,msg,a2)

#
#
#

def prinf(msg,a,len):
#
#    prints out first len many elements of array (row-wise)
#
    if (prini.numf==0):
        return

    numpy.set_printoptions(
        formatter={'int': formatf},
        linewidth=prini.lw)

    a2=numpy.array([a])
    a2=a2.flatten()
    a2=a2[0:len]

    s=sys.stdout
    f = open(prini.name,'a')

    print_main(f,msg,a2)
    print_main(s,msg,a2)

#
#
#

def formatf(x):
    return format(x,' < 8n')

#
#
#

def prini(numf,init):
#
#    (numf is the output file number)
#
    fname='out' + str(numf)

    if (init==1):
        f = open(fname,'w')
        prini.init=1

    linewidth=90

    prini.numf=numf
    prini.name=fname
    prini.lw=linewidth

    numpy.set_printoptions(threshold=sys.maxsize)

#
#
#

def prinr2(msg,a,m,n):
#
#    prints out first m-by-n submatrix
#
    if (prini.numf==0):
        return

    numpy.set_printoptions(
        formatter={'float': format2},
        linewidth=prini.lw)

#
#    copy to smaller array
#
    a2=numpy.zeros([m,n])
    for i in range(0,m):
        for j in range(0,n):
            a2[i,j]=a[i,j]
    a2=numpy.array([a2])

    s=sys.stdout
    f = open(prini.name,'a')

    print_main(f,msg,a2)
    print_main(s,msg,a2)

#
#
#

def prin2(msg,a,len):
#
#    prints out first len many elements of array (row-wise)
#
    if (prini.numf==0):
        return

    numpy.set_printoptions(
        formatter={'float': format2},
        linewidth=prini.lw)

    a2=numpy.array([a])
    a2=a2.flatten()
    a2=a2[0:len]

    s=sys.stdout
    f = open(prini.name,'a')

    print_main(f,msg,a2)
    print_main(s,msg,a2)

#
#
#

def format2(x):
    return format(x,' < 12.4E')

#
#
#

def prin2_long(msg,a,len):
#
#    prints out first len many elements of array (row-wise)
#
    if (prini.numf==0):
        return

    numpy.set_printoptions(
        formatter={'float':format2_long},
        linewidth=prini.lw)
#
###    a2=numpy.array(a,ndmin=1)
    a2=numpy.array([a])
    a2=a2.flatten()
    a2=a2[0:len]

    s=sys.stdout
    f = open(prini.name,'a')

    print_main(f,msg,a2)
    print_main(s,msg,a2)

#
#
#

def format2_long(x):
    return format(x,' < 17.8E')

#
#
#

def prinz(msg,a,len):
    if (prini.numf==0):
        return

    numpy.set_printoptions(
        formatter={'complexfloat':formatz},
        linewidth=prini.lw)

    a2=numpy.array([a])
    a2=a2.flatten()
    a2=a2[0:len]

    s=sys.stdout
    f = open(prini.name,'a')

    print_main(f,msg,a2)
    print_main(s,msg,a2)

#
#
#

def formatz(x):
    return format(x,' < 12.4E')

#
#
#

def print_main(f,msg,a):
    astring = numpy.str(a)
    astring = astring.replace('[',' ').replace(']','')
    astring = textwrap.dedent(astring)
###    fullstr=msg + '\n' + astring + '\n'

    msg='  '+msg
    f.write(msg + '\n')
    for item in astring.split('\n'):
        f.write('     '+ item)
        f.write('\n')

    f.write('\n')

#
#
#

def fill88(a,n):
    for i in range(0,n):
        a[i]=i+1

#
#
#

def bubble20(x,n):
    for ijk in range(0,2*n):
        print(ijk)
        nswaps=0

        for i in range(0,n-1):
            if x[i+1] < x[i]:
                aa=x[i+1]
                x[i+1]=x[i]
                x[i]=aa
                nswaps=nswaps+1
        if (nswaps == 0):
            break

    print('ijk=\n   ',ijk)
    print('nswaps=\n   ',nswaps)

    print('x=\n   ',x)

#
#
#

def fill_ones(x,n):
    for i in range(0,n):
        x[i]=1

#
#
#

def add17(x,y,z):
    print('Goodbye')
    z=x+y
    y=10
    return y,z
