import cv2,cv
import numpy as np
import math
import PIL
from PIL import Image
import pygame
from time import sleep

def myrange(start,end,step):
    while start>=end:
        yield start
        start -=step
#imgFile=cv.CaptureFromCam(0)
imgFile= cv.LoadImage('F:\Intel1python\Picused\intel1.jpg')
img = cv.CreateImage((400,400), 8, 3)
cv.Resize(imgFile,img)
cv.NamedWindow("Input Image",1)
cv.ShowImage('Input Image',img)
cv.WaitKey(7) % 0x100
cv.SaveImage('Saved.png',img)

sleep(3)

grey_image = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
edges = cv.CreateImage(cv.GetSize(img), cv.IPL_DEPTH_8U, 1)
cv.CvtColor(img,grey_image,cv.CV_BGR2GRAY)
cv.Smooth(grey_image, grey_image, cv.CV_GAUSSIAN, 1, 0)
cv.Canny(grey_image,edges, 80, 120)
cv.NamedWindow("Edge Detected Image",1)
cv.ShowImage('Edge Detected Image',edges)
cv.WaitKey(7) % 0x100
sleep(3)

ce = cv.CreateImage(cv.GetSize(edges), 8, 1)
#cv.CvtColor(edges, coloredges, cv.CV_GRAY2BGR)

lines=0
storage = cv.CreateMemStorage(0)
lines=cv.HoughLines2(edges,storage, cv.CV_HOUGH_STANDARD,1, np.pi/180, 100, 0, 0);

for (rho, theta) in lines[:100]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (cv.Round(x0 + 1000*(-b)), cv.Round(y0 + 1000*(a)))
    pt2 = (cv.Round(x0 - 1000*(-b)), cv.Round(y0 - 1000*(a)))
    cv.Line(ce, pt1, pt2, cv.CV_RGB(255, 255, 255) , 1, 8)

cv.NamedWindow("Line Detected Image",1)
cv.ShowImage('Line Detected Image',ce)
cv.WaitKey(7) % 0x100
cv.SaveImage('Edges.jpg',ce)
sleep(3)

h=w=400
i=1;
f1=f2=f3=f4=0;
for j in range(1,h):
    a=cv.Get2D(ce,i,j)
    if(a[0]==255.0):
      f1=1 #hori axis 2 images
      break;
j=1;
for i in range(w):
    a=cv.Get2D(ce,i,j)
    if(a[0]==255.0):
        f2=1
        break
i=w-1;
for j in range(1,h):
    a=cv.Get2D(ce,i,j)
    if(a[0]==255.0):
     f3=1
     break
j=h-1
for i in range(w):
    a=cv.Get2D(ce,i,j)
    if(a[0]==255.0):
        f4=1
        break
print f1,f2,f3,f4
if(f1==1 and f2==1 and f3==1 and f4==1):
 cas=1 #4 images
elif(f1==1 and f3==1):
 cas=3 #2 images in vert axis
elif(f2==1 and f4==1):
 cas=2#2 images in hori axis
else:
 cas=4
print cas


if(cas==1):
    i=1
    for j in range(1,h):
        a= cv.Get2D(ce,i,j)
        if(a[0]==255.0):
            x=float(j);
            break;
    print x

    i=w-1
    for j in range(1,h):
        a=cv.Get2D(ce,i,j)
        if(a[0]==255):
             x1=float(j);
             break;

    print x1
    if(x>x1):
        p=(float(x-x1)/float(w))
        print p
        p1=math.degrees(math.atan(p))
        print p1
        if p1<5:
            p1=0
        im1 = Image.open('Edges.jpg')
        imx= Image.open('Saved.png')
        imxx=imx.rotate(p1)
        imxx.save('Saved.png')
        im2=im1.rotate(p1)
        im2.save("RES.png")

    else:
        p=(float(x1-x)/float(w))
        print p
        p1=math.degrees(math.atan(p))
        print p1
        if p1<5:
         p1=0
        im1 = Image.open('Edges.jpg')
        imx= Image.open('Saved.png')
        imxx=imx.rotate(-p1)
        imxx.save('Saved.png')
        im2=im1.rotate(-p1)
        im2.save("RES.jpg")
    rotated=cv.LoadImage("RES.png")
    cv.NamedWindow("Rotated Image",1)
    cv.ShowImage('Rotated Image',rotated)
    cv.WaitKey(7) % 0x100
    sleep(3)
    
    im2 = Image.open('C:\Python27\RES.png').convert('LA')
    i=1
    y=[]
    for j in range(1,h):
        a,b=im2.getpixel((i,j))
        if(a==255):
           y.append(j) 
    print y
    cy= y[0]+ float((y[1]-y[0]))/float(2)
    ky=float((y[1]-y[0]))/float(2)
    j=10
    x=[]
    for i in range(1,w):
        a,b=im2.getpixel((i,j))
        if(a==255):
            x.append(i)
    print x
    cx= x[0]+float((x[1]-x[0]))/float(2)
    kx=float((x[1]-x[0]))/float(2)
    print cx,cy

    
    np1=cx*cy
    np2=(w-cx)*cy
    np3=(w-cx)*(w-cy)
    np4=cx*(w-cy)
    original = Image.open('Saved.png').convert('LA')

    if(np1>np2 and np1>np3 and np1>np4):
        reg=1
        print reg
        original.crop((0,0,int(cx-kx),int(cy-ky))).save("CROPPED1.png")
    elif(np2>np1 and np2>np3 and np2>np4):
        reg=2
        original.crop((int(cx+kx),0,int(w),int(cy-ky))).save("CROPPED1.png")
    elif(np3>np1 and np3>np2 and np3>np4):
        reg=3
        original.crop((int(cx+kx),int(cy+ky),int(w),int(h))).save("CROPPED1.png")
    else:
        reg=4
        original.crop((0,int(cy+ky),int(cx-kx),int(h))).save("CROPPED1.png")
    print reg
elif(cas==2):
    j=1
    for i in range(1,w):
        a= cv.Get2D(ce,i,j)
        #print a[0]
        if(a[0]==255.0):
            #print j
            x=float(i);
            break;
    print x

    j=h-1
    for i in range(1,w):
        a=cv.Get2D(ce,i,j)
        if(a[0]==255):
             x1=float(i);
             break;

    print x1
    if(x>x1):
        p=(float(x-x1)/float(h))
        print p
        p1=math.degrees(math.atan(p))
        print p1
        if p1<5:
            p1=0
        im1 = Image.open('Edges.jpg')
        imx= Image.open('Saved.png')
        imxx=imx.rotate(p1)
        imxx.save('Saved.png')
        im2=im1.rotate(p1)
        im2.save("RES.png")
        #sleep(1)
    else:
        p=(float(x1-x)/float(h))
        print p
        p1=math.degrees(math.atan(p))
        print p1
        if p1<5:
         p1=0
        im1 = Image.open('Edges.jpg')
        imx= Image.open('Saved.png')
        imxx=imx.rotate(-p1)
        imxx.save('Saved.png')
        im2=im1.rotate(-p1)
        im2.save("RES.png")
    rotated=cv.LoadImage("RES.png")
    cv.NamedWindow("Rotated Image",1)
    cv.ShowImage('Rotated Image',rotated)
    cv.WaitKey(7) % 0x100
    sleep(3)
    
    im2 = Image.open('C:\Python27\RES.png').convert('LA')
    i=1
    y=[]
    for j in range(1,h):
        a,b=im2.getpixel((i,j))
        if(a==255):
           y.append(j) 
    print y
    cy= y[0]+ float((y[1]-y[0]))/float(2)
    ky=float((y[1]-y[0]))/float(2)

    np1= w*cy
    np2= w*(h-cy)
    original = Image.open('Saved.png').convert('LA')

    if(np1>np2):
        reg=1
        print reg
        original.crop((0,0,int(w),int(cy-ky))).save("CROPPED1.png")
    else:
        reg=2
        print reg
        original.crop((0,int(cy+ky),int(w),int(h))).save("CROPPED1.png")
    
elif(cas==3):
    i=1
    for j in range(1,h):
        a= cv.Get2D(ce,i,j)
        if(a[0]==255.0):
            x=float(j);
            break;
    print x

    i=w-1
    for j in range(1,h):
        a=cv.Get2D(ce,i,j)
        if(a[0]==255):
             x1=float(j);
             break;

    print x1
    if(x>x1):
        p=(float(x-x1)/float(w))
        print p
        p1=math.degrees(math.atan(p))
        print p1
        if p1<5:
            p1=0
        im1 = Image.open('Edges.jpg')
        imx= Image.open('Saved.png')
        imxx=imx.rotate(p1)
        imxx.save('Saved.png')
        im2=im1.rotate(p1)
        im2.save("RES.png")

    else:
        p=(float(x1-x)/float(w))
        print p
        p1=math.degrees(math.atan(p))
        print p1
        if p1<5:
         p1=0
        im1 = Image.open('Edges.jpg')
        imx= Image.open('Saved.png')
        imxx=imx.rotate(-p1)
        imxx.save('Saved.png')
        im2=im1.rotate(-p1)
        im2.save("RES.jpg")
    rotated=cv.LoadImage("RES.png")
    cv.NamedWindow("Rotated Image",1)
    cv.ShowImage('Rotated Image',rotated)
    cv.WaitKey(7) % 0x100
    sleep(3)
    
    im2 = Image.open('C:\Python27\RES.png').convert('LA')
    j=10
    x=[]
    for i in range(1,w):
        a,b=im2.getpixel((i,j))
        if(a==255):
            x.append(i)
    print x
    cx= x[0]+float((x[1]-x[0]))/float(2)
    kx=float((x[1]-x[0]))/float(2)
    np1=cx*h
    np2=(w-cx)*h
    original = Image.open('Saved.png').convert('LA')

    if(np1>np2):
        reg=1
        print reg
        original.crop((0,0,int(w-cx),int(h))).save("CROPPED1.png")
    else:
        reg=2
        print reg
        original.crop((int(cx+kx),0,int(w),int(h))).save("CROPPED1.png")
else:
    original = Image.open('Saved.png').convert('LA')
    original.save("CROPPED1.png")
        
    
    
cropped = Image.open("CROPPED1.png")
cropped=cv.LoadImage("CROPPED1.png")
cv.NamedWindow("Cropped Image",1)
cv.ShowImage('Cropped Image',cropped)
cv.WaitKey(7) % 0x100
sleep(3)

ext = ".png"
imageFile="F:\\Intel1python\\Picused\\ariel.png"
im5=Image.open(imageFile)
im5 = im5.resize((400,400), PIL.Image.ANTIALIAS)
im5.save("RES1" + ext)
im6 = Image.open('C:\Python27\RES1.png').convert('LA')
im6.save("ARIELG"+ext)

imageFile="F:\\Intel1python\\Picused\\cindrella.png"
im5=Image.open(imageFile)
im5 = im5.resize((400,400), PIL.Image.ANTIALIAS)
im5.save("RES2" + ext)
im6 = Image.open('C:\Python27\RES2.png').convert('LA')
im6.save("CINDG"+ext)

imageFile="F:\\Intel1python\\Picused\\aladdin.png"
im5=Image.open(imageFile)
im5 = im5.resize((400,400), PIL.Image.ANTIALIAS)
im5.save("RES3" + ext)
im6 = Image.open('C:\Python27\RES3.png').convert('LA')
im6.save("ALADG"+ext)


imageFile="F:\\Intel1python\\Picused\\lionking.png"
im5=Image.open(imageFile)
im5 = im5.resize((400,400), PIL.Image.ANTIALIAS)
im5.save("RES4" + ext)
im6 = Image.open('C:\Python27\RES4.png').convert('LA')
im6.save("LIONG"+ext)

def featureextract (imgg,templateg):
    surfDetector = cv2.FeatureDetector_create("SURF")
    surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")

    kp = surfDetector.detect(imgg)
    kp, descritors = surfDescriptorExtractor .compute(imgg,kp)

    samples = np.array(descritors)
    responses = np.arange(len(kp),dtype = np.float32)

    knn = cv2.KNearest()
    knn.train(samples,responses)

    keys = surfDetector.detect(templateg)
    keys, desc = surfDescriptorExtractor .compute(templateg,keys)

    rowsize = len(desc) / len(keys)
    if rowsize > 1:
        hrows = np.array(desc, dtype = np.float32).reshape((-1, rowsize))
        nrows = np.array(descritors, dtype = np.float32).reshape((-1, rowsize))
    else:
        hrows = np.array(desc, dtype = np.float32)
        nrows = np.array(descritors, dtype = np.float32)
        rowsize = len(hrows[0])

    matched=0
    total=0
    for h,des in enumerate(desc):
        des = np.array(des,np.float32).reshape((1,rowsize))
        retval, results, neigh_resp, dists = knn.find_nearest(des,1)
        res,dist =  int(results[0][0]),dists[0][0]

        if dist<0.1:
            color = (0,0,255)
            matched += 1
        else:  
            color = (255,0,0)
            total+=1
        #Draw matched key points on original image
        #x,y = kp[res].pt
        #center = (int(x),int(y))
        #cv2.circle(imgg,center,2,color,-1)

        #Draw matched key points on template image
        #x,y = keys[h].pt
        #center = (int(x),int(y))
        #cv2.circle(templateg,center,2,color,-1)
    diff=matched-total
    cv2.namedWindow("Matched Keypoints in original",1);
    cv2.imshow('Matched Keypoints in original',imgg)
    cv2.namedWindow("Matched Keypoints in extracted",2);
    cv2.imshow('Matched Keypoints in extracted',templateg)
    print matched
    print total
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return diff

template = cv2.imread('C:\Python27\CROPPED1.png')

img =cv2.imread('C:\Python27\ARIELG.png')
arf=featureextract(img,template)


img =cv2.imread('C:\Python27\CINDG.png')
cinf=featureextract(img,template)


img =cv2.imread('C:\Python27\LIONG.png')
lionf=featureextract(img,template)


img =cv2.imread('C:\Python27\ALADG.png')
aladf=featureextract(img,template)

pygame.init()
screen = pygame.display.set_mode((650,400))
pygame.mixer.quit()

if(arf>cinf and arf>lionf and arf>aladf):
  movie = pygame.movie.Movie("D:\\ariel.mpg")
elif(cinf>lionf and cinf>aladf):
  movie = pygame.movie.Movie("D:\\cinderella.mpg")
elif(lionf>aladf):
  movie = pygame.movie.Movie("D:\\lionking.mpg")
else:
    movie = pygame.movie.Movie("D:\\aladdin.mp4")

 # sh=Image.open('C:\Python27\RES4.png')
flag=0

if (movie.has_audio()):
    print ("movie has audio")
movie.set_volume (1.0)
movie.play()
while True:
    if not(movie.get_busy()):
        print("rewind")
        movie.rewind()
        movie.play()

    for event in pygame.event.get():
        if event.type==pygame.MOUSEBUTTONUP and event.button==1 and flag==0:
            print "stop"
            movie.stop()
            flag=1
            sleep(15)
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and flag==1:
            print "resume"
            movie.play()
            flag=0
 
    if pygame.QUIT in [e.type for e in pygame.event.get()]:
            break





    
    
    



    
    
        
