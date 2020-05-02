import cv2
import math

videoFilePath = 'Train Tom and jerry.mp4'
saveFolder = 'Train/'
cap = cv2.VideoCapture(videoFilePath)
frameRate = cap.get(5)

i=0
count = 0
while(cap.isOpened()):
    cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    count += math.floor(frameRate)
    print(count)

    ret, frame = cap.read()
    # cv2.imshow('frame',frame)
    if (ret != True) or cv2.waitKey(1) & 0xFF == ord('q') or count>=8912:
        break

    cv2.imwrite(saveFolder+"frame"+str(i)+".jpg",frame)
    i+=1
    # if count % math.floor(frameRate) == 0:

cap.release()
cv2.destroyAllWindows()
