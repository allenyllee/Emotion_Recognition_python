import cv2, glob, random, math, numpy as np, dlib, itertools
from sklearn.svm import SVC
from sklearn.externals import joblib


#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file
emotions = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"] #Emotion list
def get_landmarks2(xlist,ylist):
    xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
    ymean = np.mean(ylist)
    xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
    ycentral = [(y-ymean) for y in ylist]

    if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
        anglenose = 0
    else:
        anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

    if anglenose < 0:
        anglenose += 90
    else:
        anglenose -= 90

    landmarks_vectorised = []
    for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
        landmarks_vectorised.append(x)
        landmarks_vectorised.append(y)
        meannp = np.asarray((ymean,xmean))
        coornp = np.asarray((z,w))
        dist = np.linalg.norm(coornp-meannp)
        anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
        landmarks_vectorised.append(dist)
        landmarks_vectorised.append(anglerelative)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised
    
    
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        xmean = np.mean(xlist) #Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist] #get distance between each point and the central point in both axes
        ycentral = [(y-ymean) for y in ylist]

        if xlist[26] == xlist[29]: #If x-coordinates of the set are the same, the angle is 0, catch to prevent 'divide by 0' error in function
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(x)
            landmarks_vectorised.append(y)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    if len(detections) < 1: 
        landmarks_vectorised = "error"
    return landmarks_vectorised
    
    

clf = joblib.load('filename.pkl') #filename2.pk1

while True:
    image = cv2.imread("anger_Image.jpg") #open image
    #ret, frame = video_capture.read()
    frame=image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For each detected face
        #print(d)
        ##image = cv2.imread("anger_Image.jpg") #open image
        
        #clf.predict(landmarks_vectorised)
        
        shape = predictor(clahe_image, d) #Get coordinates
        xlist = []
        ylist = []
        for i in range(1,68): #There are 68 landmark points on each face
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        #print(len(xlist))
        landmarks_vectorised=get_landmarks2(xlist,ylist)
        #print(landmarks_vectorised)
        matriz3 = np.array(landmarks_vectorised)
        #print (type(matriz3))
        print(matriz3.shape)
        print (type(landmarks_vectorised))
        #print(matriz3)
        matriz_R=matriz3.reshape(1, -1)
        print(matriz_R.shape)
        #print(matriz_R)
        Solution=clf.predict(matriz_R)
        SS=int(Solution)
        print(emotions[SS])
        font = cv2.FONT_HERSHEY_SIMPLEX
    ##cv2.putText(frame,emotions[SS], ((xlist[19]),(ylist[19]), font, 1,(255,255,255),2)
        cv2.putText(frame,emotions[SS], (int(xlist[19]),int(ylist[19])), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow("image", frame) #Display the frame

    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break



