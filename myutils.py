import glob
import time
import cv2
squashImage = True
import numpy as np
def cropImage(img,rect):
    return img[rect[1]:rect[3],rect[0]:rect[2],:]
def chunks(l, n):
    n = max(1, n)
    return [l[i:i + n] for i in range(0, len(l), n)]
def boxCenters(boxes):
    x = (boxes[:,0:1]+boxes[:,2:3])/2
    y = (boxes[:,1:2]+boxes[:,3:4])/2
    return concatenate((x,y),1)

def plotRectangle_img(img,t,color=(1,0,0),thickness=3):
    t = np.asarray(t).astype(int)
    cv2.rectangle(img,tuple(t[0:2]),tuple(t[2:4]),color,thickness)

def showWindowsOnImages(res,color=(1,0,0)):
    nfiles = len(res.keys())
    f = figure(1,figsize=(10,13))
    clf()
    mm = ceil(sqrt(float(nfiles)))
    i = 0
    for f,b in res.iteritems():
        h = b
        #fc6 = b[1]
        f = os.path.join(imgBaseDir,f)
        sys.stdout.write('.')
        img = caffe.io.load_image(f)
        if squashImage:
            img = cv2.resize(img,(227,227))
        subplot(mm,mm,i+1)
        i = i+1
        curImg = img.copy()
        t = h.astype(int)
        plotRectangle_img(curImg,t,color = color)
        #cv2.rectangle(curImg,tuple(t[0:2]),tuple(t[2:4]),(1,0,0),3)
        imshow(curImg)
        h = reshape(h,(1,4))
        curCenters = boxCenters(h)
        #scatter(curCenters[:,0],curCenters[:,1],c='r')
    draw()
def sampleTiledRects(img_size,boxSizeRatio=[5],boxOverlap=.5):
    if not isinstance(boxSizeRatio,list):
        boxSizeRatio = [boxSizeRatio]
    all_h = []
    for sizeRatio in boxSizeRatio:
        boxSize = np.asarray(img_size[0:2])/sizeRatio # height, width
        b = boxSize*(1-boxOverlap)
        topLefts = [(x,y) for x in arange(0,img_size[1]-boxSize[1],b[1]) for y in arange(0,img_size[0]-boxSize[0],b[0])]
        xs,ys = zip(*topLefts)
        xs = floor(np.asarray(xs))
        ys = floor(np.asarray(ys))
        all_h.append(vstack([xs,ys,xs+boxSize[1],ys+boxSize[0]]).T)
    return vstack(all_h)
def sampleRandomRects(img_size,boxSizeRatio=5,nBoxes=50):
    boxSize = int(mean(img_size[0:2])/boxSizeRatio)
    center_ys = np.asarray(randint(low=boxSize/2,high=img_size[0]-boxSize/2,size=nBoxes))
    center_xs = np.asarray(randint(low=boxSize/2,high=img_size[1]-boxSize/2,size=nBoxes))
    h = vstack([center_xs-boxSize/2,center_ys-boxSize/2,center_xs+boxSize/2,center_ys+boxSize/2]).T
    return h,zip(center_xs,center_ys)
def sampleWindows(files,boxSizeRatio,boxOverlap,random_set = 0):
    if not isinstance(boxSizeRatio,list):
        boxSizeRatio = [boxSizeRatio]
    res = []
    rects = []
    imgIndices = []
    for i,fn in enumerate(files):

        fn = os.path.join(imgBaseDir,fn)
        curImg = caffe.io.load_image(fn)
        if squashImage:
            curImg = cv2.resize(curImg,(227,227))
        img_size = curImg.shape
        for sizeRatio in boxSizeRatio:
            curRects = sampleTiledRects(img_size,boxSizeRatio=sizeRatio,boxOverlap=boxOverlap)
            if random_set > 0:
                numpy.random.shuffle(curRects)
                curRects = curRects[:random_set]
            curRects = around(curRects)
            curWindows = [cropImage(curImg,r) for r in curRects]
            res.extend(curWindows)
            rects.extend(curRects)
            imgIndices.extend(curRects.shape[0]*[i])
    return res,rects,imgIndices
def sampleRandomWindows(files,boxSizeRatio,boxOverlap,nPerImage=5):
    a,b,c = sampleWindows(files,boxSizeRatio,boxOverlap,random_set=nPerImage)
    return a,b,c
def boxIntersection(b1,b2):
    xmin1 = b1[0]
    xmax1 = b1[2]
    xmin2 = b2[0]
    xmax2 = b2[2]
    ymin1 = b1[1]
    ymax1 = b1[3]
    ymin2 = b2[1]
    ymax2 = b2[3]
    res = np.asarray([max(xmin1,xmin2),max(ymin1,ymin2),min(xmax1,xmax2),min(ymax1,ymax2)])
    return res
def boxArea(b):
    if b[2] < b[0]:
        return 0
    if b[3] < b[1]:
        return 0
    return (b[2]-b[0])*(b[3]-b[1])
def boxesOverlap(boxes1,boxes2):
    boxes1 = reshape(np.asarray(boxes1),(-1,4))
    boxes2 = reshape(boxes2,(-1,4))
    n1 = boxes1.shape[0]
    n2 = boxes2.shape[0]
    #print n1
    #print n2
    res = zeros((n1,n2))

    for i1,b1 in enumerate(boxes1):
        b1Area = boxArea(b1)
        for i2,b2 in enumerate(boxes2):
            curInt = boxIntersection(b1,b2)
            intArea = boxArea(curInt)
            if intArea<=0:
                continue
            b2Area = boxArea(b2)
            assert(b1Area>0)
            assert(b2Area>0)
            assert(b1Area+b2Area-intArea > 0)
            res[i1,i2] = float(intArea)/(b1Area+b2Area-intArea)
    return res
def boxDims(b):
    width = b[2]-b[0]
    height = b[3]-b[1]
    return width,height
def boxAspectRatio(b):
    width,height = boxDims(b)
    width = float(width)
    height = float(height)
    res = max([height/width,width/height])
    return res
def inflateBox(b,f):
    width,height = boxDims(b)
    width = (width*f)/2
    height = (height*f)/2
    center_x = (b[0]+b[2])/2
    center_y = (b[1]+b[3])/2
    #print center_x,center_y
    res = [center_x-width,center_y-height,center_x+width,center_y+height]
    return res
def chopLeft(box,p,mode=0):
    if mode == 0:
        box[0] = (1-p)*box[0]+p*box[2]
    else:
        box[0] = box[0]+p
    return box
def chopRight(box,p,mode=0):
    if mode == 0:
        box[2] = (1-p)*box[2]+p*box[0]
    else:
        box[2] = box[2]-p
    return box
def chopTop(box,p,mode=0):
    if mode == 0:
        box[1] = (1-p)*box[1]+p*box[3]
    else:
        box[1] = box[1]+p
    return box
def chopBottom(box,p,mode=0):
    if mode == 0:
        box[3] = (1-p)*box[3]+p*box[1]
    else:
        box[3] = box[3]-p
    return box

def chopAll(box,p,mode=0):
    box = chopBottom(chopTop(chopLeft(chopRight(box,p,mode),p,mode),p,mode),p,mode)
    return box
def plotRectangle(r,color='r'):
    if isinstance(r,ndarray):
       # print 'd'
        r = r.tolist()
   # print r
    points = [[r[0],r[1]],[r[2],r[1]],[r[2],r[3]],[r[0],r[3]],]
   # print points
    line = plt.Polygon(points, closed=True, fill=None, edgecolor=color,lw=2)
    gca().add_line(line)
    #show()
def splitBox(box,p,mode):
    return [chopLeft(box.copy(),p,mode),
            chopRight(box.copy(),p,mode),
            chopTop(box.copy(),p,mode),
            chopBottom(box.copy(),p,mode),chopAll(box.copy(),p,mode)]
            #inflateBox(box.copy(),.8,)]

def relBox2Box(box):
    box[2] = box[2]+box[0]
    box[3] = box[3]+box[1]
    return box

def boxCenter(box):
    w,h = boxDims(box)
    boxCenterX = box[0]+w/2
    boxCenterY = box[1]+h/2
    return boxCenterX,boxCenterY
def makeSquare(box):
    w,h = boxDims(box)
    m = float(max([w,h]))/2
    boxCenterX = box[0]+w/2
    boxCenterY = box[1]+h/2
    newBox = [boxCenterX-m,boxCenterY-m,boxCenterX+m,boxCenterY+m]
    return newBox
def clipBox(targetBox,box):
    targetBox = list(targetBox)
    if len(targetBox) == 2:
        targetBox = [0,0]+targetBox[::-1]
    newBox = boxIntersection(targetBox,box)
    return newBox