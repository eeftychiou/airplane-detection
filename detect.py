
import numpy as np
import cv2,os,sys
import argparse

import hashlib

from common import clock, draw_str
from collections import namedtuple



Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
#define default folders
curfolder = os.path.join(os.curdir)
masks = os.path.join(os.curdir, 'masks')
detect_outfolder = os.path.abspath(os.path.join(os.curdir, 'detect_outfolder'))
motion_outfolder =os.path.abspath(os.path.join(os.curdir, 'motion_outfolder'))
diff_outfolder = os.path.abspath(os.path.join(os.curdir, 'diff_outfolder'))
positives = os.path.abspath(os.path.join(os.curdir, 'positives'))
negatives = os.path.abspath(os.path.join(os.curdir, 'negatives'))
vidseq = os.path.abspath(os.path.join(os.curdir, 'videoseq'))

folders=["detect_outfolder","motion_outfolder","diff_outfolder","settings","negatives","positives","videoseq"]

negatives_list=os.path.join(os.curdir, "settings", "negatives.txt")
positives_list=os.path.join(os.curdir, "settings", "positives.txt")
ignore_list=os.path.join(os.curdir, "settings", "ignore.txt")

fgbg =cv2.bgsegm.createBackgroundSubtractorMOG()

#load mask
mask=cv2.imread(os.path.join(masks,"smallmask.png"))
mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)

parser = argparse.ArgumentParser(description='Process options')
parser.add_argument('--cascade', help='define cascade classifier',required=True )
parser.add_argument('--video', help='define video',required=True)
parser.add_argument('--savedetections', help='save Haar Detections', action="store_true")
parser.add_argument('--savemotiondetections', help='Save motion detections', action="store_true")
parser.add_argument('--overlapratio', help='Overlap Ratio 0-1', default=0.6)
parser.add_argument('--savediffs', help='Save detection differences. That is save all images not overlaping',  action="store_true")
parser.add_argument('--showHaarDetections', help='Show Haar Detections in the window',  action="store_true")
parser.add_argument('--showMotionDetections', help='Show Motion Detections in the window',  action="store_true")
parser.add_argument('--UpdateLists', help='Updates the positive and negative lists',  action="store_true")
parser.add_argument('--EnableMotionDetect', help='Enables Motion Detection',  action="store_true")
parser.add_argument('--EnableOutStream', help='Enables Streaming to pipe',  action="store_true")
parser.add_argument('--SaveOutStream', help='Saves Image Stream for video Creation',  action="store_true")

args = parser.parse_args()

cascade_fn = args.cascade
video_src =  args.video
namesum=hashlib.md5(video_src).hexdigest()

#Constants
contourArea=230
fogSumThresh = 15000
border=50



def create_folders():
    for folder in folders:
        if not os.path.exists(os.path.join(folder)):
            os.mkdir(os.path.join(folder))


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(20, 10), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

def save_rects(img, rects, index, folder):
    for x1, y1, x2, y2 in rects:
        imgsv=img[y1:y2,x1:x2]
        cv2.imwrite(os.path.join(folder, 'image_{}_{}{}{}{}_{}.png'.format(namesum,x1, y1, x2, y2,index)), imgsv)

def save_rect(img, rect, index, folder):
    x1, y1, x2, y2 = rect
    imgsv=img[y1:y2,x1:x2]
    cv2.imwrite(os.path.join(folder, 'image_{}_{}{}{}{}_{}.png'.format(namesum,x1, y1, x2, y2,index)), imgsv)

def areadiff(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return float(dx*dy)
    else:
        return 0

def rect_area(a):  # returns None if rectangles don't intersect
    dx = a.xmax-a.xmin
    dy = a.ymax-a.ymin
    if (dx>=0) and (dy>=0):
        return float(dx*dy)
    else:
        return 0

def motion_detect(frame):



    rects=[]
    MOGfgmask = fgbg.apply(frame)

    if MOGfgmask==None:
        return

    MOGfgmask=np.bitwise_and(mask,MOGfgmask)

    #remove the shadows
    MOGfgmask[MOGfgmask==127]=0

    contour=MOGfgmask.astype('uint8')
    # get individual objects
    contour = cv2.dilate(contour, None, iterations=6)
    #cv2.imshow("contour after dilate",contour)
    _, contours0, _  = cv2.findContours( contour.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

    image_indx = 0

    for c in contours:

        if cv2.contourArea(c) < contourArea:  # not a very good measure of the object within the area
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        fogsum=np.sum(MOGfgmask[y:y+h,x:x+w])
        if fogsum < fogSumThresh:
            continue

        #exclude edges of image 30 pixels on each side
        if (x+w)+border>frame.shape[1] or x-border < 0 or y+h+border>frame.shape[0] or y-border<0 :
            continue

        #add detection to list
        rects.append((x, y, x+w, y+h))

    image_indx=image_indx+1
    if args.savemotiondetections:
        save_rects(frame, rects, image_indx, motion_outfolder)


    #cv2.imshow('Motion frame',frame)
    return rects

def search_file(file,searchtxt):
    searchfile = open(file , "r")

    for line in searchfile:
        if searchtxt in line:
            searchfile.close()
            return True

    searchfile.close()
    return False

def create_list():

    # open files for append
    neg = open(negatives_list, "a+")

    pos = open(positives_list, "a+")

    ign = open(ignore_list, "a+")


    filenames = sorted(os.listdir(diff_outfolder))

    for filename in filenames:

        if search_file(positives_list,filename) or search_file(negatives_list,filename) or search_file(ignore_list,filename):
            continue

        img=cv2.imread(os.path.join(diff_outfolder,filename))

        cv2.imshow("candidate",img)
        w,h=img.shape[1], img.shape[0]
        k = cv2.waitKey(0)

        if k == ord('p'):
            linestr = os.path.join(diff_outfolder, filename + " 1 {} {} {} {}\n".format(0, 0 , w, h))
            pos.write(linestr)
            continue

        elif k == ord('n'):
            linestr = os.path.join(diff_outfolder, filename +"\n")
            neg.write(linestr)
            continue
        elif k== ord("d"):
            linestr = os.path.join(diff_outfolder, filename +"\n")
            ign.write(linestr)
            continue
        elif k == 27:
            break
        else:
            continue

    pos.close()
    neg.close()



if __name__ == '__main__':
    import sys, getopt



    index=0

    cam = cv2.VideoCapture(video_src)
    create_folders()

    cascade = cv2.CascadeClassifier(cascade_fn)


    while True:
        ret, img = cam.read()

        if ret==False:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        if args.EnableMotionDetect:
            motion_rects=motion_detect(img)
        vis = img.copy()
        visclean = img.copy()

        dt = clock() - t
        if args.showHaarDetections:
            draw_rects(vis, rects, (0, 255, 0))

        if args.showMotionDetections:
            draw_rects(vis, motion_rects, (255,0 , 0))
            draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))


        if args.EnableOutStream:
            cv2.imwrite("testpipe.png",vis)
            #sys.stdout.write( vis.tostring() )
        else:
            cv2.imshow('planedetect', vis)

        if args.SaveOutStream:
            cv2.imwrite(os.path.join(vidseq,"frame_{:08}.png".format(index)),vis)

        if args.savediffs:
            if len(rects)==0 or len(motion_rects)==0: #no detection of Haar or motion
                save_rects(visclean,motion_rects,index, diff_outfolder)
                save_rects(visclean,rects,index, diff_outfolder)
            else:
                for recta in motion_rects:
                    for rectb in rects:
                        a=Rectangle(*recta[0:4])
                        b=Rectangle(*rectb[0:4])
                        if abs(areadiff(a,b)/rect_area(a))<=args.overlapratio:
                            save_rect(visclean,recta,index, diff_outfolder)
                            save_rect(visclean,rectb,index, diff_outfolder)

        if args.savedetections:
            save_rects(visclean,rects,index, detect_outfolder)


        index=index+1



        if 0xFF & cv2.waitKey(5) == 27:
            break

        #cv2.waitKey(0)

    cv2.destroyAllWindows()

    if args.UpdateLists:
        create_list()
