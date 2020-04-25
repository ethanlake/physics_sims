import argparse
import matplotlib
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import matplotlib.animation as animation
from scipy.linalg import expm

matplotlib.rc('font', family='serif')

I = np.identity(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
sig = np.array([X,Y,Z])

ZZ = np.kron(Z,Z)
XI = np.kron(X,I)
IX = np.kron(I,X)
ZI = np.kron(Z,I)
IZ = np.kron(I,Z)
YI = np.kron(Y,I)
IY = np.kron(I,Y)
II = np.kron(I,I)

def Hvec(J,h,vec): #just acts with the hamiltonian on the input vector
    return np.dot(II-deltat*1j*(J*ZZ-h*(XI+IX)),vec.Ts)

def dsum(a,b):
    mat = np.zeros(np.add(a.shape,b.shape))
    mat[:a.shape[0],:a.shape[1]]=a
    mat[a.shape[0]:,a.shape[1]:]=b
    return mat

parser = argparse.ArgumentParser(description="let's make some generative art!!")
# add arguments
parser.add_argument('-steps', dest='steps', required=False)
parser.add_argument('-N', dest='N', required=False)
parser.add_argument('-J', dest='J', required=False)
parser.add_argument('-field', dest='field', required=False)
parser.add_argument('-zfield', dest='zfield', required=False)
parser.add_argument('-fname', dest='fname', required=False)
parser.add_argument('-deltat', dest='deltat', required=False)
args = parser.parse_args()

deltat = 0.05

N = 2
if args.N:
    N = int(args.N)

J = 1
if args.J:
    J = float(args.J)

h = 0
if args.field:
    h = float(args.field)

hz = 0
if args.zfield:
    hz = float(args.zfield)

bins = 110

### build some matrices! ###
### very dumb way of doing things XD ###
ZZs = []
Zs = []
Xs = []
Ys = []

for i in range(0,N):
    if i == 0:
        addtoXs = X
        addtoZs = Z
        addtoYs = Y
        addtoZZs = Z
    else:
        if(i==N-1):
            addtoZZs = Z
        else:
            addtoZZs = I
        addtoZs = I
        addtoXs = I
        addtoYs = I
    for j in range(1,N):
        if(j==i):
            addtoXs = np.kron(addtoXs,X)
            addtoZs = np.kron(addtoZs,Z)
            addtoYs = np.kron(addtoYs,Y)
            addtoZZs = np.kron(addtoZZs,Z)
        else:
            addtoXs = np.kron(addtoXs,I)
            addtoYs = np.kron(addtoYs,I)
            addtoZs = np.kron(addtoZs,I)
            if(j==i+1):
                addtoZZs = np.kron(addtoZZs,Z)
            else:
                addtoZZs = np.kron(addtoZZs,I)

    Xs.append(addtoXs)
    Ys.append(addtoYs)
    Zs.append(addtoZs)
    ZZs.append(addtoZZs)


### initialize wavefunction ###
psi = np.array([1,0])
for i in range(1,N):
    if i == (N-1)/2:
        psi = np.kron(psi,np.array([0,1]))
    else:
        psi = np.kron(psi,np.array([1,0]))

print("starting wavefunction = ",psi)

Hamiltonian = np.zeros((2**N,2**N))
for i in range(0,N):
    Hamiltonian = Hamiltonian - J * ZZs[i] - h * Xs[i] - hz * Zs[i]

print("starting H = ", Hamiltonian)

def vev(vec,mat):
    return np.dot(np.dot(np.conj(vec),mat),vec.T)

thetarange = np.linspace(0,np.pi/2,bins)
phirange = np.linspace(0,2*np.pi,bins)
thetas, phis = np.meshgrid(thetarange, phirange) #rectangular plot of polar data

xs = np.sin(thetas) * np.cos(phis)
ys = np.sin(thetas) * np.sin(phis)
zs = np.cos(thetas)

vevs = np.zeros((N,bins,bins))

steps = 1
if args.steps:
    steps = int(args.steps)

fname = "test"
deltat = .07
if args.deltat:
    deltat = float(deltat)
if args.fname:
    fname = args.fname

Ut = expm(-1j * Hamiltonian * deltat) #will need to put this inside the loop if you want to do H(t) problems

for s in range(0,steps):
    print("on step %i..."%s)
    fig, axes = plt.subplots(1,N, subplot_kw=dict(polar=True),figsize=(2*N, 2))
    for i in range(0,N):
        axes[i].axis("off")

    for i in range(0,bins):
        for j in range(0,bins):
            for k in range(0,N):
                vevs[k,i,j] = vev(psi,Xs[k]*xs[i,j] + Ys[k]*ys[i,j] + Zs[k]*zs[i,j])

    for i in range(0,N):
        axes[i].pcolormesh(phis, thetas, vevs[i,:]) #X,Y & data2D must all be same dimensions

    psi = np.dot(Ut,psi) #update the wavefunction --- we're in Schrodinger picture

    if(steps!=1):
        plt.savefig(fname+"%.3i"%s, dpi=None, facecolor='k', edgecolor='k',
        orientation='portrait', bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close()
    else:
        print("showing plot...")
        plt.show()
        break

