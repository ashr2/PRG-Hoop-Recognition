import cv2
import os
vidcap = cv2.VideoCapture('../assets/IMG_8620.MOV')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("../assets/training_data/image"+str(count)+".jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 0.2 #//it will capture image in each 0.5 second
count=len(os.listdir("../assets/training_data"))
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)