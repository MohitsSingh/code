import glob
import time
import cv2
squashImage = True

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
    t = asarray(t).astype(int)    
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