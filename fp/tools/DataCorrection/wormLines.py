"""
Miguel Taylor (mtlazul@gmail.com)
License: this code is in the public domain
"""
import re, cv2, math, json
import numpy as np
import networkx as nx
from scipy import interpolate
from scipy.spatial import Delaunay

# counter clockwise
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    if np.all(A == D) or np.all(B == C):
        return False
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def hasIntersections(points):
    for i in range(0,len(points)-1):
        for j in range(i,len(points)-1):
            A = points[i]
            B = points[i+1]
            C = points[j]
            D = points[j+1]
            if intersect(A,B,C,D):
                return True
    return False
    

# Computes the index of array where you should
# insert an element to minimize the Euclidean
# distance with its neighbours
def minDistIndex(array, element):    
    dist1 = dist2 = float('inf')
    i1 = i2 = 0
    for i in range(0, len(array)):
        dist = np.linalg.norm(array[i]-element)
        if dist < dist1:
            dist2 = dist1
            i2 = i1
            dist1 = dist
            i1 = i
        else:
            if dist < dist2:
                dist2 = dist
                i2 = i
    if i2 < i1:
        return i2+1
    else:
        return i1+1

def getGradientSum(p1, p2, xGradient, yGradient):
    deltaX,deltaY = np.array(p2) - np.array(p1)
    vectorDir = [float(deltaX), float(deltaY)]
    if abs(deltaX) < abs(deltaY):
        vectorDir /= abs(deltaY)
        iterationEnd = abs(deltaY)
    else:
        vectorDir /= abs(deltaX)
        iterationEnd = abs(deltaX) 
    gradientAcc = 0
    for i in range(1,iterationEnd):
        ix = int(p1[0]+i*vectorDir[0])
        iy = int(p1[1]+i*vectorDir[1])
        vectGradient = np.array([xGradient[iy,ix],yGradient[iy, ix]])
        gradientAcc += (abs(np.dot(vectorDir/np.linalg.norm(vectorDir), vectGradient)))**0.5
    return gradientAcc/(0.012*iterationEnd+1)

def getDoGKernel(k, sigma):
    X = range((-k//2)+1, (k//2)+1)
    Y = []
    for x in X:
        y = math.e**(-(x**2)/sigma**2)*-2*x/sigma**2
        Y += [y]
    Y = np.reshape(Y,(k,-1))
    Y = Y / np.linalg.norm(Y)
    return Y

def minDist(points):
    mesh = Delaunay(points)
    edges = np.vstack((mesh.vertices[:,:2], mesh.vertices[:,-2:]))

    x = mesh.points[edges[:,0]]
    y = mesh.points[edges[:,1]]

    dists = (np.sum((x-y)**2, 1))**0.5
    idx = np.argmin(dists)
    return dists[idx]

def getPerpOffset(Magnitude, p1,p2):
    deltaX,deltaY = np.array(p2) - np.array(p1)
    vectorDir = [float(deltaY), -float(deltaX)]
    vectorDir =  vectorDir / np.linalg.norm(vectorDir)
    return Magnitude*vectorDir

class landmarks():
    def __init__(self, data):
	#Match the head attribute 
        headMatch = re.findall('head \((\d*) (\d*)', data)
        self.head = np.array(map(int,headMatch[0]))
	
	#Match the tail attribute
        tailMatch = re.findall('tail \((\d*) (\d*)', data)
        self.tail = np.array(map(int,tailMatch[0]))
	
        #Match the rest of the points
        #first 2 matches correspond to head and tail, thus are ignored
        pointMatch = re.findall('\((\d*) (\d*)\)', data)
        points = []
        for i in range(2, len(pointMatch)):
            points.append(map(int,pointMatch[i]))
        self.points = np.array(points)
        

    def fix(self, xGradient, yGradient):
        self.fixInvertedLandmarks()
        self.fixHeadTailpossition()
        self.fixInvertedLandmarks()
        if hasIntersections(self.points):
            newPoints = self.fixLandmarksOrder(xGradient, yGradient)
            if hasIntersections(newPoints):
                pass
            else:
                self.points = newPoints
        else:
            if ccw(self.points[1], self.points[0], self.points[-2]):
                self.points = self.points[::-1]
        self.updateTailPosition()

    def updateTailPosition(self):
        tailMatch = np.where(np.all(self.points == self.tail, axis=1))
        if len(tailMatch[0]) > 0:
            self.tailIndex = tailMatch[0][0]
        else:
            self.tailIndex = len(self.points)//2
            
        
    def fixLandmarksOrder(self, xGradient, yGradient):

        # Get the graph from Delaunay transform
        # weight is proportional to the gradient between 2 points
        G = self.getDelaunayGraph(xGradient, yGradient)


        # Head is the first node and vertex
        node = 0
        vertices = [0]
        edges = sorted(G[node], key=lambda x: G[node][x]['weight'])
        
        
        # Get the second node following clockwise order
        p0 = self.points[node]
        p1 = self.points[edges[0]]
        p2 = self.points[edges[1]]
        G.remove_node(node)
        if ccw(p2, p0, p1):
            node = edges[0]
            vertices += [edges[0]]
        else:
            node = edges[1]
            vertices += [edges[1]]
        edges = sorted(G[node], key=lambda x: G[node][x]['weight'])


        # while not tail
        while not np.all(self.points[node] == self.tail) and len(edges)>0:
            p0 = self.points[node]
            v0 = np.array(self.points[node])-np.array(self.points[vertices[-2]])
            candidates = []
            for i in range(0,min(3, int(round(len(edges)/3.0)+1))):
                pi = self.points[edges[i]]
                vi = np.array(pi)-np.array(p0)
                vi = vi / np.linalg.norm(vi)
                candidates += [[i, np.dot(vi, v0), edges[i]]]
            candidates = np.array(candidates)
            chosen = int(candidates[candidates[:,1].argsort()[::-1]][0,0])
            G.remove_node(node)
            node = edges[chosen]
            vertices += [edges[chosen]]
            edges = sorted(G[node], key=lambda x: G[node][x]['weight'])

        # after tail
        while len(edges)>0:
            G.remove_node(node)
            node = edges[0]
            vertices += [edges[0]]
            edges = sorted(G[node], key=lambda x: G[node][x]['weight'])

        # reorder points
        return self.points[vertices+[0]]

    def getDelaunayGraph(self, xGradient, yGradient):
        G = nx.Graph()
        G.add_nodes_from(range(0,len(self.points[0:-1])))
        tri = Delaunay(self.points[0:-1])
        # for each Delaunay triangle add 3 edges to the graph
        # the 'weight' of each edge is determined by the gradient sum between 2 points 
        for n in xrange(tri.nsimplex):
            edge = [tri.vertices[n,0], tri.vertices[n,1]]
            weight = getGradientSum(self.points[edge[0]],self.points[edge[1]], xGradient, yGradient)
            distance = np.linalg.norm(self.points[edge[1]]-self.points[edge[0]])
            G.add_edge(edge[0], edge[1],weight=weight)
            edge = sorted([tri.vertices[n,0], tri.vertices[n,2]])
            weight = getGradientSum(self.points[edge[0]],self.points[edge[1]], xGradient, yGradient)
            distance = np.linalg.norm(self.points[edge[1]]-self.points[edge[0]])
            G.add_edge(edge[0], edge[1], weight=weight)
            edge = sorted([tri.vertices[n,1], tri.vertices[n,2]])
            weight = getGradientSum(self.points[edge[0]],self.points[edge[1]], xGradient, yGradient)
            distance = np.linalg.norm(self.points[edge[1]]-self.points[edge[0]])
            G.add_edge(edge[0], edge[1], weight=weight)
        return G

    # fix inverted landmarks by calculating if inverting a pair of
    # landmarks reduces the overal distance of those landmarks with their
    # neighbours
    def fixInvertedLandmarks(self):
        for i in range(0, len(self.points)-3):
            dist01 = np.linalg.norm(self.points[i]-self.points[i+1])
            dist02 = np.linalg.norm(self.points[i]-self.points[i+2])
            dist31 = np.linalg.norm(self.points[i+3]-self.points[i+1])
            dist32 = np.linalg.norm(self.points[i+3]-self.points[i+2])
            if (dist02+dist31) < (dist01+dist32):
                self.points[[i+1,i+2],:] = self.points[[i+2,i+1],:]

    # deletes the tail and head from the points array and then find the best
    # possition to insert them back
    def fixHeadTailpossition(self):
        #Delete the head from the points array
        headPos = np.where(np.all(self.points == self.head, axis=1))
        self.points = np.delete(self.points, headPos, axis=0)
        #Find the best possition for the head
        index = minDistIndex(self.points, self.head)
        self.points = np.concatenate((self.points[:index],[self.head],self.points[index:]), axis=0)
        #In the points array: [0] and [-1] = head
        while not(np.array_equal(self.points[0],self.head)):
            self.points = np.append(self.points[1:],[self.points[0]],axis=0)
        if not (np.array_equal(self.head,self.points[-1])):
            self.points = np.append(self.points,[self.head],axis=0)
        #Delete the tail from the points array
        tailPos = np.where(np.all(self.points == self.tail, axis=1))
        self.points = np.delete(self.points, tailPos, axis=0)
        #Find the best possition for the tail
        index = minDistIndex(self.points, self.tail)
        self.points = np.concatenate((self.points[:index],[self.tail],self.points[index:]), axis=0)
                
                

class nematode():
    #light constructor for batch processing
    def __init__(self, imageRoute, dataRoute, nPoints):
        self.nPoints = nPoints
        self.imageRoute = imageRoute
        self.dataRoute = dataRoute
        self.landmarks = landmarks(open(dataRoute,"r").read())
        self.image = cv2.imread(imageRoute, cv2.IMREAD_GRAYSCALE)
        self.getXYGradient()
        self.landmarks.fix(self.xGradient, self.yGradient)
        self.points = np.copy(self.landmarks.points)

    def __init__(self, imageRoute, dataRoute, axes, nPoints):
        self.nPoints = nPoints
        self.imageRoute = imageRoute
        self.dataRoute = dataRoute
        self.landmarks = landmarks(open(dataRoute,"r").read())
        self.image = cv2.imread(imageRoute, cv2.IMREAD_GRAYSCALE)
        self.getXYGradient()
        self.landmarks.fix(self.xGradient, self.yGradient)
        self.points = np.copy(self.landmarks.points)
        self.splineA = self.getSpline(self.points, 0, self.landmarks.tailIndex)                                      
        self.splineB = self.getSpline(self.points, self.landmarks.tailIndex, len(self.landmarks.points)-1)
        self.axes = axes
        self.plot()

    def getXYGradient(self):
        self.scale = int(minDist(self.landmarks.points))*2
        DoG = getDoGKernel(self.scale, (self.scale+1)/6.0)
        Gauss = cv2.getGaussianKernel(self.scale, (self.scale+1)/6.0)
        self.xGradient = cv2.sepFilter2D(self.image, cv2.CV_32F, DoG, Gauss)
        self.yGradient = cv2.sepFilter2D(self.image, cv2.CV_32F, Gauss, DoG)

    def shiftLandmarkPosition(self, point, shamt):
        index = np.where(np.all(self.landmarks.points == point, axis=1))[0][0]

        #check boudaries
        if index+shamt < 1 or index+shamt > len(self.landmarks.points)-2 :
            return

        #swap elements
        temp = np.copy(self.landmarks.points[index])
        self.landmarks.points[index] = self.landmarks.points[index+shamt]
        self.landmarks.points[index+shamt] = temp

        #update data
        self.landmarks.updateTailPosition()
        self.points = np.copy(self.landmarks.points)
        self.splineA = self.getSpline(self.points, 0, self.landmarks.tailIndex)
        self.splineB = self.getSpline(self.points, self.landmarks.tailIndex, len(self.landmarks.points)-1)

    # plots the landmarks and the splines
    def plot(self):
        rows,cols = self.image.shape
        self.axes.imshow(self.image,cmap = 'gray', interpolation='nearest', picker=False)
        points = np.transpose(self.points)
        spline = np.concatenate((self.splineA,self.splineB),axis=1)
        self.pointPlot, = self.axes.plot(points[0], points[1], 'o', picker=5)
        self.annotationList = []
        for label, x, y in zip(range(len(points[0])-1), points[0], points[1]):
            offset = getPerpOffset(self.scale, [points[0][label-1], points[1][label-1]],[points[0][label+1], points[1][label+1]])
            annotation = self.axes.annotate(label,xy=(x, y), xytext=(offset[0], -offset[1]),
                                            textcoords='offset points', ha='right', va='bottom',
                                            bbox=dict(boxstyle='round,pad=0.05', fc='white', alpha=0.5),
                                            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            self.annotationList.append(annotation)
        self.splinePlot, = self.axes.plot(spline[0], spline[1], 'r', picker=False)
        self.axes.axis([0, cols, rows, 0])
        self.axes.axis('off')

    # updates the plot data
    def updatePlot(self):
        points = np.transpose(self.points)
        spline = np.concatenate((self.splineA,self.splineB),axis=1)
        self.pointPlot.set_xdata(points[0])
        self.pointPlot.set_ydata(points[1])
        self.splinePlot.set_xdata(spline[0])
        self.splinePlot.set_ydata(spline[1])
        for annotation in self.annotationList:
            annotation.remove()
        self.annotationList = []
        for label, x, y in zip(range(len(points[0])-1), points[0], points[1]):
            offset = getPerpOffset(self.scale, [points[0][label-1], points[1][label-1]],[points[0][label+1], points[1][label+1]])
            annotation = self.axes.annotate(label,xy=(x, y), xytext=(offset[0], -offset[1]),
                                            textcoords='offset points', ha='right', va='bottom',
                                            bbox=dict(boxstyle='round,pad=0.05', fc='white', alpha=0.5),
                                            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            self.annotationList.append(annotation)
        
    # start:closed end:closed
    def getSpline(self, pointArray, start, end):
        points = pointArray[start:end+1]
        points = np.transpose(points)
        x,y = [points[0], points[1]]
        if end-start < 3:
            xnew,ynew = [x,y]
        else:
            tck,u = interpolate.splprep([x,y], ub = 1, ue = -2)
            xnew,ynew = interpolate.splev( np.linspace( 0, 1, self.nPoints ), tck,der = 0)
        return [xnew, ynew]

    # Tries to find a better fitting for the landmarks
    # Alpha determines the maximum movement allowed (in pixels)
    def adjustLandmarks(self, alpha):
        for i in range(1,len(self.landmarks.points)-1):
            maxGradient = 0
            currentPoint = newPoint = self.landmarks.points[i]
            deltaX,deltaY = self.landmarks.points[i+1] - self.landmarks.points[i-1]
            vectorDir = [float(-deltaY), float(deltaX)]
            #normalize vectorDir
            if abs(deltaX) < abs(deltaY):
                vectorDir /= abs(deltaY)
            else:
                vectorDir /= abs(deltaX)
            for a in range(-alpha,alpha):
                ix = int(currentPoint[0]+a*vectorDir[0])
                iy = int(currentPoint[1]+a*vectorDir[1])
                if self.insideImage([[ix,iy]]):
                    vectGradient = np.array([self.xGradient[iy,ix],self.yGradient[iy, ix]])
                    dirGradient = abs(np.dot(vectorDir/np.linalg.norm(vectorDir), vectGradient))
                    if (dirGradient > maxGradient):
                        maxGradient = dirGradient
                        newPoint = np.array([ix,iy])
            self.points[i] = newPoint
        self.splineA = self.getSpline(self.points, 0, self.landmarks.tailIndex)
        self.splineB = self.getSpline(self.points, self.landmarks.tailIndex, len(self.points)-1)

    # Tries to find a better fitting for the spline
    # Alpha determines the maximum movement allowed (in pixels)
    def adjustSpline(self, alpha):
        #calculate spline points
        newSplineA = self.getSpline(self.points, 0, self.landmarks.tailIndex)
        newSplineB = self.getSpline(self.points, self.landmarks.tailIndex, len(self.points)-1)
        newSpline = np.transpose(np.concatenate((newSplineA, newSplineB),axis=1))
        pointIndex = 0
        for i in range(0,len(newSpline)):
            if pointIndex < len(self.points)-2:
                #calculate distance to landmarks
                dist1 = np.linalg.norm(newSpline[i] - self.points[pointIndex])
                dist2 = np.linalg.norm(newSpline[i] - self.points[pointIndex+2])
                if dist2 < dist1:
                    pointIndex += 1
            
            maxGradient = 0
            currentPoint = newSpline[i]
            deltaX,deltaY = self.points[pointIndex] - self.points[pointIndex+1]
            vectorDir = [float(-deltaY), float(deltaX)]
            #normalize vectorDir
            if abs(deltaX) < abs(deltaY):
                vectorDir /= abs(deltaY)
            else:
                vectorDir /= abs(deltaX)
            for a in range(-alpha,alpha):
                ix = int(currentPoint[0]+a*vectorDir[0])
                iy = int(currentPoint[1]+a*vectorDir[1])
                if self.insideImage([[ix,iy]]):
                    vectGradient = np.array([self.xGradient[iy,ix],self.yGradient[iy, ix]])
                    dirGradient = abs(np.dot(vectorDir/np.linalg.norm(vectorDir), vectGradient))
                    if (dirGradient > maxGradient):
                        maxGradient = dirGradient
                        newSpline[i] = np.array([ix,iy])
        self.splineA = self.getSpline(newSpline, 0, len(newSpline)//2-1)
        self.splineB = self.getSpline(newSpline, len(newSpline)//2, len(newSpline)-1)

    def getMeanGradienLine(self, p1, p2):
        deltaX,deltaY = p2 - p1
        vectorDir = [float(deltaX), float(deltaY)]
        if abs(deltaX) < abs(deltaY):
            vectorDir /= abs(deltaY)
            iterationEnd = abs(deltaY)
        else:
            vectorDir /= abs(deltaX)
            iterationEnd = abs(deltaX) 
        gradientAcc = 0 
        for i in range(1,iterationEnd):
            ix = int(p1[0]+i*vectorDir[0])
            iy = int(p1[1]+i*vectorDir[1])
            vectGradient = np.array([self.xGradient[ix,iy],self.yGradient[ix, iy]])
            gradientAcc += abs(np.dot(vectorDir/np.linalg.norm(vectorDir), vectGradient))
        return gradientAcc/iterationEnd

    # returns true if every point in the input array is inside the
    # boundaries of the image
    def insideImage(self, points):
        rows,cols = self.image.shape
        for i in range(0,len(points)):
            row = points[i][0]
            col = points[i][1]
            if row < 0 or col < 0 or row > rows-1 or col > cols-1:
                return False
        return True

    def getNematodeSize(self, splineA, splineB):
        n = len(splineA)-2
        p = splineA[0]
        length = 0
        if n <= 0:
            avrWidth = 0
            VarWidth = 0
        else:
            widthAcc = 0
            widthAccSq = 0
            midPoints = (splineA+splineB)/2
            for i in range(1,len(splineA)-1):
                length += np.linalg.norm(p-midPoints[i])
                p = midPoints[i]
                width = np.linalg.norm(splineA[i]-splineB[i])
                widthAcc += width
                widthAccSq += width**2
            avrWidth = widthAcc / n;
            VarWidth = widthAccSq / n - avrWidth * avrWidth;
        length += np.linalg.norm(p-splineA[-1])
        return (avrWidth, math.sqrt(VarWidth), length)

    # returns the serialized json
    def serialize(self):
        splineA = np.transpose(self.splineA).astype(int)
        splineB = np.transpose(self.splineB).astype(int)
        (avrWidth, stdWidth, length) = self.getNematodeSize(splineA, splineB[::-1])
        spline = np.concatenate((splineA, splineB), axis=0)
        data = {"imageRoute": self.imageRoute,
                "dataRoute": self.dataRoute,
                "spline": spline.tolist(),
                "size":{"widthMean":avrWidth,
                        "widthStandardDeviation":stdWidth,
                        "length":length},
                "landmarks": {"head": self.landmarks.head.tolist(),
                              "tail": self.landmarks.tail.tolist(),
                              "points": self.landmarks.points.tolist()}}
        return json.dumps(data)
