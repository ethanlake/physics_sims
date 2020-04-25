import argparse
import matplotlib
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import matplotlib.animation as animation
from scipy.linalg import expm

def commutator(A,B):
    return np.dot(A,B) - np.dot(B,A)

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
    #return np.dot(-J*ZZ-h*(XI+IX),vec.T)
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
        psi = np.kron(psi,np.array([1,0]))
    else:
        psi = np.kron(psi,np.array([1,0]))

#psi = np.random.rand(2**N)
#psi = psi / norm(psi)
print("starting wavefunction = ",psi)

Hamiltonian = np.zeros((2**N,2**N))
for i in range(0,N):
    Hamiltonian = Hamiltonian - J * ZZs[i] - h * Xs[i] - hz * Zs[i]

print("starting H = ", Hamiltonian)

def otoc(vec,A,B): #want A to be the time-evolved version of some operator and B to be fixed
    uwuv = np.dot(np.dot(np.dot(A,B),A),B)
    return (1-np.dot(np.dot(np.conj(vec),uwuv),vec.T).real)/2

thetarange = np.linspace(0,np.pi/2,bins)
phirange = np.linspace(0,2*np.pi,bins)
thetas, phis = np.meshgrid(thetarange, phirange) #rectangular plot of polar data

xs = np.sin(thetas) * np.cos(phis)
ys = np.sin(thetas) * np.sin(phis)
zs = np.cos(thetas)

otocs = np.zeros((int(round(N+1)/2),bins,bins)) #taking advantage of translation symmetry to realize that the otoc is only a function of the absolute value of the distance between the two operators --- therefore for N = 2l + 1, there are only l+1 distinct otocs

steps = 1
if args.steps:
    steps = int(args.steps)


fname = "test"
deltat = .04
if args.deltat:
    deltat = float(deltat)
if args.fname:
    fname = args.fname

Wt = Zs[int(round((N-1)/2))]
Ut = expm(-1j*Hamiltonian*deltat) #only need to calculate this ONCE! jesus christ dude
time = 0

for s in range(0,steps):
    time = time + deltat
    print("step %i, time %.3f"%(s,time))
    fig, axes = plt.subplots(1,N, subplot_kw=dict(polar=True),figsize=(2*N, 2))
    for i in range(0,N):
        axes[i].axis("off")

    for i in range(0,bins):
        for j in range(0,bins):
            for k in range(0,int(round(N+1)/2)):
                otocs[k,i,j] = otoc(psi,xs[i,j]*Xs[k]+ys[i,j]*Ys[k]+zs[i,j]*Zs[k],Wt)

    #print("max otoc: %.3f, min otoc: %.3f"%(np.max(otocs),np.min(otocs)))

    for i in range(0,int(round(N+1)/2)-1):
        axes[i].pcolormesh(phis, thetas, otocs[i,:],vmin=0,vmax=1)
        axes[(-i-1)%N].pcolormesh(phis, thetas, otocs[i,:],vmin=0,vmax=1)
    axes[i+1].pcolormesh(phis, thetas, otocs[i+1,:],vmin=0,vmax=1) #do the center plot

    Wt = np.dot(np.dot(np.conj(Ut.T),Wt),Ut) #evolve the matrix forward by one timestep

    if(steps!=1):
        plt.savefig(fname+"%.3i"%s, dpi=None, facecolor='k', edgecolor='k',
        orientation='portrait', bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close('all')
    else:
        plt.show()
        break



'''
u, v = np.mgrid[0:np.pi:10j, 0:2*np.pi:10j]
x = np.sin(u) * np.cos(v)
y = np.sin(u) * np.sin(v)
z = np.cos(u)



def Vev(vec,mat):
    return np.dot(np.dot(np.conj(vec),mat),vec.T)

def update(vec,J,h):
    vevs1 = np.zeros((10,10))
    vevs2 = np.zeros((10,10))
    for i in range(0,np.shape(x)[0]):
        for j in range(0,np.shape(x)[1]):
            vevs1[i,j] = (Vev(vec,XI*x[i,j]) + Vev(vec,YI*y[i,j]) + Vev(vec,ZI*z[i,j])).real
            vevs2[i,j] = (Vev(vec,IX*x[i,j]) + Vev(vec,IY*y[i,j]) + Vev(vec,IZ*z[i,j])).real

    newVec = (np.dot(II,vec.T) - 1j * deltat * Hvec(J,h,vec)).T

    return vevs1, vevs2, newVec

initvec = np.kron(np.array([1j,0]),np.array([1j,0]))
vec = initvec

expectationVals1, expectationVals2, vec = update(initvec,J,h)

norm = colors.Normalize(vmin = -1,
                      vmax = 1, clip = False)


img1 = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,
                       facecolors=cm.coolwarm(norm(expectationVals1)))

img2 = ax2.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.Spectral,
                       linewidth=0, antialiased=False,
                       facecolors=cm.Spectral(norm(expectationVals2)))

print("running animation... ")
img2.set_facecolors((1,1,1,1))


steps = 0
if args.steps:
    steps = int(args.steps)
if steps==0:
    def animate(frameNum,img1,img2,J,h,vec):
        originalcm = cm.Spectral(norm(expectationVals2[:]))
        expectationVals1[:], expectationVals2[:], vec[:] = update(vec,J,h)
        newcm = cm.Spectral(norm(expectationVals2))
        print(originalcm - newcm)

        vec = vec / np.sqrt(np.dot(np.conj(vec).T,vec))

        img1.set_facecolors(np.ndarray.tolist(cm.Spectral(norm(expectationVals1)))) #colors[vals[k]])
        img2.set_facecolors(np.ndarray.tolist(cm.Spectral(norm(expectationVals2))))

    anim = animation.FuncAnimation(fig, animate, fargs=(img1,img2,J,h,vec), frames=10, interval=15, blit=False)


plt.show()
'''
