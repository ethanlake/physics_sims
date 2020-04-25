import argparse
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import scipy.integrate as integrate
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.collections import LineCollection
import math as math


def ndelta(n,a,b):
	if a == b: #potts
		return 1
	else:
		return 0

def ndist(n,a,b):
	return min((a-b)%n,(b-a)%n) #soft XYy thing

def update(vals,neighbors,numFlips,N,J,mu):
	newVals = vals.copy()
	for i in range(0,numFlips): #probably just take numFlips = 1 unless you want to change up the animation settings --- else keep numFlips small enough so that we don't have to copy the values everytime to prevent flip conflicts
		flipSite = np.random.randint(0,np.shape(vals)[0]) #choose a site to be flipped in mcmc
		flipSign = np.random.randint(0,2)*2-1 #get random sign for the flip
		deltaE = 0
		for j in range(0,len(neighbors[flipSite])): #add up the energy difference for flipping that site
			deltaE += J * (ndelta(N,(vals[flipSite]+flipSign)%N,vals[neighbors[flipSite][j]]) -  ndelta(N,vals[flipSite],vals[neighbors[flipSite][j]]))
		deltaE += mu * (ndist(N,(vals[flipSite]+flipSign)%N,0) - ndist(N,vals[flipSite],0))
		if deltaE < 0:
			newVals[flipSite] = (vals[flipSite] + flipSign)%N
			#print("flipped from %i to %i with energy %f"%(vals[flipSite], (vals[flipSite]+flipSign)%N,deltaE))
		else:
			if(np.random.rand(1) <= np.exp(-deltaE)):
				newVals[flipSite] = (vals[flipSite] + flipSign)%N
				#print("flipped from %i to %i with energy %f"%(vals[flipSite], (vals[flipSite]+flipSign)%N,deltaE))
		#newVals[flipSite] = (vals[flipSite] + flipSign)%N
	#print(newVals)
	return newVals

fig, ax = plt.subplots()
plt.axis("off")
xbnds = np.array([0,13])
ybnds = np.array([0,16])

xbf, ybf = 1, 1

xplt = xbnds + np.array([xbf,-xbf])
yplt = ybnds + np.array([ybf,-ybf])
ax.set_xlim(xplt)
ax.set_ylim(yplt)

parser = argparse.ArgumentParser(description="let's make some generative art!!")
# add arguments
parser.add_argument('-numPts', dest='numPts', required=False)
parser.add_argument('-N', dest='N', required=False)
parser.add_argument('-J', dest='J', required=False) #AF if positive
parser.add_argument('-nx', dest='nx', required=False)
parser.add_argument('-ny', dest='ny', required=False)
parser.add_argument('-steps', dest='steps', required=False)
parser.add_argument('-thermalSteps', dest='thermalizationTime',required=False)
parser.add_argument('-mu', dest='mu', required=False)
parser.add_argument('-frames', dest='frames', required=False)
parser.add_argument('-fps', dest='fps',required=False)
parser.add_argument('-numFlips', dest='numFlips',required=False)
parser.add_argument('-hide', dest='hide',required=False)
args = parser.parse_args()

numPts = 50
if args.numPts:
	numPts = int(args.numPts)

numFlips = int(round(numPts/20))
if args.numFlips:
	numFlips = int(args.numFlips)

N = 4
if args.N:
	N = int(args.N)

J = 1
if args.J:
	J = float(args.J)
mu = 0
if args.mu:
	mu = float(args.mu)

nx = 10
ny = 10

if args.nx:
	nx = int(args.nx)
if args.ny:
	ny = int(args.ny)

## generate the voronoi tiling
#or, make a nice sort-of-random but more uniform distribution
if(args.nx and args.ny):
	xs = np.linspace(xbnds[0],xbnds[1],nx)
	ys = np.linspace(ybnds[0],ybnds[1],ny)

	randstr = .7
	pts = np.zeros((2,nx*ny))
	k=0
	for i in range(0,nx):
		for j in range(0,ny):
			pts[0,k] = xs[i] + randstr * np.random.rand(1) * (xbnds[1]-xbnds[0]) / nx
			pts[1,k] = ys[j] + randstr * np.random.rand(1) * (ybnds[1]-ybnds[0]) / ny
			k += 1
	numPts = k
	pts = pts.transpose()

else:
	x = np.random.uniform(*xbnds,size=numPts).reshape((numPts,1))
	y = np.random.uniform(*ybnds,size=numPts).reshape((numPts,1))

	pts = np.hstack([x,y])  #a 2xnumPts grid with all the xy coords of the points



#create the graph!!
vor = Voronoi(pts)
vertCoords = vor.vertices #these are the coordinate values of the voronoi points
numVerts = np.shape(vertCoords)[0]
polyVertLabels = vor.regions #the labels of the points bounding each polygon in the voronoi diagram (integers) -- each element in the array is a list of point labels
ridgeVertLabels = vor.ridge_vertices #labels of the voronoi points (integers)
numEdges = np.shape(ridgeVertLabels)[0]
dualVerts = vor.point_region #index of the voronoi region for each input point; maps input points to polygons

goodPolyVertLabels = [s+s[0:1] for s in polyVertLabels if len(s)>0  and -1 not in s] #get the list of the "good" polygons
#goodPolyVertLabels = [s for s in polyVertLabels if len(s)>0]

polyCoords = np.array([vertCoords[s] for s in goodPolyVertLabels]) #coordinate values of the vertices on the boundaries of each good polygon --- polygons[i] gives an nx2 matrix, where n is the number of vertices on the boundary of the polygon
numPolys = np.shape(polyCoords)[0] #just look at the good ones

#now get the dual edges to those in the voronoi tiling
dualEdges = np.zeros((numEdges,2),dtype=int)
edgeCount = 0

print("establishing dual graph: number of good polygons = %i"%numPolys)
for i in range(0,numPolys):
	for j in range(i+1,numPolys):
		commonVerts = np.intersect1d(goodPolyVertLabels[i],goodPolyVertLabels[j])
		if(np.shape(commonVerts)[0] == 2):
			dualEdges[edgeCount,0] = i
			dualEdges[edgeCount,1] = j
			edgeCount+=1 #edge count keeps track of only those edges that are on the boundary between two good polygons
dualEdges = dualEdges[:edgeCount,:] #some of the edges lead off to infinity so here we cut down to only the edges in the interior

#now that we have a list of all dual edges connecting the polygons, we want to make a list of the nn's of each polygon, since we'll be using this to do the mcmc step
nearestNeighbors = np.zeros((numPolys),dtype=object) #will be a tuple where each entry is a list containing the nearest neighbors of that polygon (restricted to good polygons)
for i in range(0,numPolys): #note: using nearestNeighbors.fill([]) screws stuff up
	nearestNeighbors[i] = []
for e in range(0,edgeCount):
	v1 = dualEdges[e,0]
	v2 = dualEdges[e,1]
	nearestNeighbors[v1].append(v2)
	nearestNeighbors[v2].append(v1)

#for i in range(0,numpolys): #could add something to fill cells by color of vertices or something
#    polyinds = verts[shapes_ind[i]]
    #if(np.random.rand(1) > .5):
	#	ax.fill(polyinds[:,0],polyinds[:,1],"b")

#initialize the array of values
vals = np.zeros((numPolys))
vals = np.random.randint(0,N,numPolys)

#run some hidden stuff to thermalize if desired

print("doing initial thermalization...")
if args.thermalizationTime:
	for i in range(0,int(args.thermalizationTime)):
		vals = update(vals,nearestNeighbors,numFlips,N,J,mu)

print("ready to start!")
print("parameters are: J=%.2f, mu=%.2f, N=%i"%(J,mu,N))



#####################################
#### do the plotting and animation
#####################################

colors=[(143/255, 3/255, 47/255), #wine
(26/255, 6/255, 156/255), #deep blue
(174/255, 255/255, 158/255), #light glass green
(219/255, 160/255, 42/255), #lightish orange
(75/255, 201/255, 140/255)] #turqoiseish

colors=[(0.6534409842368321, 0.04144559784698193, 0.2668204536716648, 1.0),
(0.6196078431372549, 0.00392156862745098, 0.25882352941176473, 1.0),
(0.6280661284121491, 0.013302575932333718, 0.26082276047673975, 1.0),
(0.6449826989619377, 0.03206459054209919, 0.2648212226066898, 1.0),
(0.6365244136870435, 0.022683583237216455, 0.26282199154171476, 1.0)
]
#cmap = plt.cm.gnuplot
#cmap = plt.cm.tab20c #nice pastels
cmap = plt.cm.Set3 #very light pastels
#cmap = plt.cm.Set2
cmap = plt.cm.Spectral #very stained-glass vibe


filledPolys = []
for i in range(0,numPolys):
	polygon = patches.Polygon(polyCoords[i],closed=True, ec='k',lw=2,fc=cmap(vals[i]/N))
	filledPolys.append(polygon)
	ax.add_patch(polygon)
	#ax.fill(polyCoords[i][:,0],polyCoords[i][:,1],c=cmap(vals[i]),alpha=1)

steps = 0
if args.steps:
	steps = int(args.steps)
if steps==0:
	def animate(frameNum):
		vals[:] = update(vals[:],nearestNeighbors,numFlips,N,J,mu)
		for k in range(numPolys):
			filledPolys[k].set_facecolor(cmap(vals[k]/(N))) #colors[vals[k]])
		return filledPolys,

	anim = animation.FuncAnimation(fig, animate,
                               frames=10, interval=15, blit=False)


'''
#ax.scatter(ridges[:,0], ridges[:,1])
#initialize colors, then loop through links --- then learn how to do metropolis updates at some temperature --- and make a graphical widget that allows you to slide the temperature while the animation is running :--0

#plt.plot(verts[:,0], verts[:, 1], 'ko', ms=8)
zordercount = 0
for vpair in ridgeVertLabels:
	if vpair[0] >= 0 and vpair[1] >= 0:
		v0 = vor.vertices[vpair[0]]
		v1 = vor.vertices[vpair[1]]
		# Draw a line from v0 to v1.
		plt.plot([v0[0], v1[0]], [v0[1], v1[1]], 'k', linewidth=2, zorder=zordercount)
		zordercount+=1
'''

#plt.scatter(vertCoords[:,0], vertCoords[:,1], c=vals,cmap='Paired', linewidth=2,marker="h",s=100,zorder=zordercount)
if not args.hide:
	plt.show()
