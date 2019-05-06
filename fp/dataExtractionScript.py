import os, glob, json, cv2, math
from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import time
from random import randint

def getDoGKernel(k, sigma):
    X = range((-k//2)+1, (k//2)+1)
    Y = []
    for x in X:
        y = math.e**(-(x**2)/sigma**2)*-2*x/sigma**2
        Y += [y]
    Y = np.reshape(Y,(k,-1))
    Y = Y / np.linalg.norm(Y)
    return Y

def getXYGradient(img, scale):
    DoG = getDoGKernel(scale, (scale+1)/6.0)
    Gauss = cv2.getGaussianKernel(scale, (scale+1)/6.0)
    xGradient = cv2.sepFilter2D(img, cv2.CV_32F, DoG, Gauss)
    yGradient = cv2.sepFilter2D(img, cv2.CV_32F, Gauss, DoG)
    return [xGradient, yGradient]


#load the data
path = 'json/'
data = [] 
for filename in glob.glob(os.path.join(path, '*.json')):
    data_file = open(filename)
    data += [json.load(data_file)]


# examples of data access
# print len(data)
# print data[0]["imageRoute"]
# print len(data[0]["spline"])
# print data[0]["size"]["length"]
# print data[0]["size"]["widthStandardDeviation"]
# print data[0]["size"]["widthMean"]


# get the mean width
width = 0
for d in data:
    width += d["size"]["widthMean"]
width /= len(data) 
print "Mean Width: "+str(int(width))


# Number of lines for training and testing
nLines = 30

# %train
trainP = 0.8

# Length of the lines
lenLines = 64


X_train = np.array([], dtype='float32').reshape(0,lenLines,lenLines)
Y_train = np.array([], dtype='float32').reshape(0,lenLines,lenLines)
X_test = np.array([], dtype='float32').reshape(0,lenLines,lenLines)
Y_test = np.array([], dtype='float32').reshape(0,lenLines,lenLines)

for i,d in enumerate(data):
    print "data #"+str(i)+"/"+str(len(data))
    num = i
    #try:
    spline = np.array(d["spline"])
    imgRoute = d["imageRoute"].replace("/media/jjimenez/Seagate Expansion Drive/nematodos", "/home/mtlazul/Documents")
    print (imgRoute)
    img = cv2.imread(imgRoute, cv2.IMREAD_GRAYSCALE)
    rows,cols = img.shape
    
    # mask: nematode 255, background 0
    img2 = np.zeros(img.shape)
    cv2.fillPoly(img2,[spline],(255) )

    #plt.imshow(img2, cmap = 'gray', interpolation='nearest')

    # ----------------NEMATODE-------------------

    try:
        tck,u = interpolate.splprep([spline[0:19,0],spline[0:19,1]], ub = 1, ue = -2)
        newSplineA = np.transpose(interpolate.splev( np.linspace( 0, 1, nLines//2+2), tck,der = 0))
        tck,u = interpolate.splprep([spline[20:39,0],spline[20:39,1]], ub = 1, ue = -2)
        newSplineB = np.transpose(interpolate.splev( np.linspace( 0, 1, nLines//2+2), tck,der = 0))
    except:
        print("Skipping nematode, something went wrong in the transposed")
        continue
    
    X = np.array([], dtype='float32').reshape(0,lenLines, lenLines)
    Y = np.array([], dtype='float32').reshape(0,lenLines, lenLines)
    for i in range(1,len(newSplineA)-1):
        currentPoint = newSplineA[i]
        deltaX,deltaY = newSplineA[i+1] - newSplineA[i-1]
        vectorDir = [float(-deltaY), float(deltaX)]
        #normalize vectorDir
        if abs(deltaX) < abs(deltaY):
            vectorDir /= abs(deltaY)
        else:
            vectorDir /= abs(deltaX)
        #ixa = []
        #iya = []
        for offset in range(-(lenLines//4)+1, (lenLines//2)-1,4):
            x = np.array([], dtype='float32')
            y = np.array([], dtype='float32')
            offset_value=randint(-(lenLines//4)+1, (lenLines//2))
            ix = int(round(currentPoint[0]+(offset_value)*vectorDir[0]))
            iy = int(round(currentPoint[1]+(offset_value)*vectorDir[1]))
            if iy < 0 or ix < 0 or iy > rows-1 or ix > cols-1:
                break
            x = np.array(img[iy-(lenLines//2):iy+(lenLines//2),ix-(lenLines//2):ix+(lenLines//2)],dtype='float32')
            y = np.array(img2[iy-(lenLines//2):iy+(lenLines//2),ix-(lenLines//2):ix+(lenLines//2)],dtype='float32')
            if x.shape == (lenLines,lenLines) and not np.all(y==0) and y.shape == (lenLines,lenLines):
                X = np.concatenate((X,[x]))
                Y = np.concatenate((Y,[y]))

    for i in range(1,len(newSplineB)-1):
        currentPoint = newSplineB[i]
        deltaX,deltaY = newSplineB[i+1] - newSplineB[i-1]
        vectorDir = [float(-deltaY), float(deltaX)]
        #normalize vectorDir
        if abs(deltaX) < abs(deltaY):
            vectorDir /= abs(deltaY)
        else:
            vectorDir /= abs(deltaX)
        #ixa = []
        #iya = []
        for offset in range(-(lenLines//4)+1, (lenLines//2)-4):
            x = np.array([], dtype='float32')
            y = np.array([], dtype='float32')
            offset_value=randint(-(lenLines//4)+1, (lenLines//2))
            ix = int(round(currentPoint[0]+(offset_value)*vectorDir[0]))
            iy = int(round(currentPoint[1]+(offset_value)*vectorDir[1]))
            if iy < 0 or ix < 0 or iy > rows-1 or ix > cols-1:
                break
            x = np.array(img[iy-(lenLines//2):iy+(lenLines//2),ix-(lenLines//2):ix+(lenLines//2)],dtype='float32')
            y = np.array(img2[iy-(lenLines//2):iy+(lenLines//2),ix-(lenLines//2):ix+(lenLines//2)],dtype='float32')
            if x.shape == (lenLines,lenLines) and not np.all(y==0) and y.shape == (lenLines,lenLines):
                X = np.concatenate((X,[x]))
                Y = np.concatenate((Y,[y]))
    # ---------------NOT NEMATODE-------------------

    imgdx,imgdy = getXYGradient(img, int(width//2))

    imgdm = (imgdx**2 + imgdy**2)**0.5

    x =  np.concatenate((newSplineA[:,0],newSplineB[:,0]))
    y =  np.concatenate((newSplineA[:,1],newSplineB[:,1]))
    GaussX = cv2.getGaussianKernel(cols, np.std(x))
    GaussY = cv2.getGaussianKernel(rows, np.std(y))
    Gauss2d = np.outer(GaussY,GaussX)

    tx = np.mean(x)- cols//2
    ty = np.mean(y)- rows//2
    M = np.float32([[1,0,tx],[0,1,ty]])
    Gauss2d = cv2.warpAffine(Gauss2d,M,(cols,rows))
    imgdm *= Gauss2d

    kernel = np.ones((4*(lenLines//2),4*(lenLines//2)), np.uint8)
    img2_dilation = cv2.dilate(img2, kernel, iterations=1)
    imgdm *= 255-img2_dilation
    imgdm /= np.max(imgdm)
    #plt.imshow(imgdm, cmap = 'gray', interpolation='nearest')
    #plt.show()
    # non maximum suppression
    indices = []
    aperture = 0.3
    while len(indices) < nLines:
        if aperture > 0.9:
            break
        timg = np.zeros((rows, cols))
        li = 0.5-aperture
        ls = 0.5+aperture
        p2s = np.where(np.all((imgdm>li,imgdm<ls),axis=0))
        p2s = np.array(zip(p2s[1], p2s[0]))
        for p in p2s:
            if p[0]-1 < 0 or p[1]-1 < 0 or p[0]+1 > rows-1 or p[1]+1 > cols-1:
                break
            p2 = imgdm[p[0], p[1]]
            angle = np.arctan2(imgdy[p[0], p[1]],imgdx[p[0], p[1]])/math.pi
            if abs(angle) <= 1/8 or abs(angle) >= 7/8:
                p1 = imgdm[p[0],p[1]-1]
                p3 = imgdm[p[0],p[1]+1]
            elif abs(angle) >= 3/8 and abs(angle) <= 5/8:
                p1 = imgdm[p[0]-1,p[1]]
                p3 = imgdm[p[0]+1,p[1]]
            elif angle > -3/8 and angle < -1/8 or angle > 5/8 and angle < 7/8:
                p1 = imgdm[p[0]+1,p[1]-1]
                p3 = imgdm[p[0]-1,p[1]+1]
            else:
                p1 = imgdm[p[0]-1,p[1]+1]
                p3 = imgdm[p[0]+1,p[1]-1]
            if p2 > p1 and p2 > p3:
                timg[p[0], p[1]] = 1
        indices = np.where(timg == 1)
        indices = np.array(zip(indices[0], indices[1]))
        aperture += 0.2
    #indices = np.array(indices)
    #print indices
    order = np.arange(indices.shape[0])
   # np.random.shuffle(order)
    for ro in range(0,nLines,10):
        try:
            currentPoint = indices[ro]
        except:
            print ("Skipping non nematode")
            break
        deltaX = imgdx[currentPoint[0], currentPoint[1]] 
        deltaY = imgdy[currentPoint[0], currentPoint[1]] 
        vectorDir = [float(-deltaY), float(deltaX)]
        #normalize vectorDir
        if abs(deltaX) < abs(deltaY):
            vectorDir /= abs(deltaY)
        else:
            vectorDir /= abs(deltaX)
        #ixa = []
        #iya = []
        for offset in range(-(lenLines//2)+1, (lenLines//2)-1,4):
            x = np.array([], dtype='float32')
            y = np.array([], dtype='float32')
            offset_value=randint(-(lenLines//4)+1, (lenLines//2))
            ix = int(round(currentPoint[0]+(offset_value)*vectorDir[0]))
            iy = int(round(currentPoint[1]+(offset_value)*vectorDir[1]))
            if iy < 0 or ix < 0 or iy > rows-1 or ix > cols-1:
                break
            x = np.array(img[iy-(lenLines//2):iy+(lenLines//2),ix-(lenLines//2):ix+(lenLines//2)],dtype='float32')
            #plt.imshow(x, cmap = 'gray', interpolation='nearest')
            #plt.show()
            y = np.array(img2[iy-(lenLines//2):iy+(lenLines//2),ix-(lenLines//2):ix+(lenLines//2)],dtype='float32')
            #plt.imshow(y, cmap = 'gray', interpolation='nearest')
            # plt.show()
            if x.shape == (lenLines,lenLines) and np.all(y==0) and y.shape == (lenLines,lenLines):
                #print x.shape
                #print y.shape
                X = np.concatenate((X,[x]))
                Y = np.concatenate((Y,[y]))
        #plt.plot(ixa,iya,'b')
    X /= 255
    Y /= 255
    order = np.arange(X.shape[0])
    np.random.shuffle(order)
    nTrain = int(X.shape[0] * trainP)
    X_train = np.concatenate((X_train,X[order[0:nTrain]]))
    Y_train = np.concatenate((Y_train,Y[order[0:nTrain]]))
    X_test = np.concatenate((X_test,X[order[nTrain:]]))
    Y_test = np.concatenate((Y_test,Y[order[nTrain:]]))
    if (num > 20):
        break


print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

X_train = X_train.reshape(X_train.shape[0], lenLines, lenLines, 1)
Y_train = Y_train.reshape(X_train.shape[0], lenLines, lenLines)
X_test = X_test.reshape(X_test.shape[0], lenLines, lenLines, 1)
Y_test = Y_test.reshape(X_test.shape[0], lenLines, lenLines)

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

np.save('X_train',X_train)
np.save('Y_train',Y_train)
np.save('X_test',X_test)
np.save('Y_test',Y_test)

