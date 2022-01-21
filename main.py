import numpy as np
import cv2
import math

# Funcion que obtiene el angulo del defecto de convexidad
def angle(s,e,f):
    v1 = [s[0]-f[0],s[1]-f[1]]
    v2 = [e[0]-f[0],e[1]-f[1]]
    ang1 = math.atan2(v1[1],v1[0])
    ang2 = math.atan2(v2[1],v2[0])
    ang = ang1 - ang2
    if (ang > np.pi):
        ang -= 2*np.pi
    if (ang < -np.pi):
        ang += 2*np.pi
    return ang*180/np.pi

# Funcion que obtiene el contorno de la mano y el hull
def getcnthull(image):
    contours, hierarchy = cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)
    return contours, hull

# Función que obtiene los defectos de convexidad
def getdefects(contours):
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull) # Almacena los defectos de convexidad
    return defects

# Funcion que cuenta los dedos
def countFingers(fingerConvexityDefects, handRect):
    numberOfFingers = len(fingerConvexityDefects)
    if numberOfFingers > 0:
        return numberOfFingers + 1
    elif (handRect[3] > handRect[2] * 1.3):
        return 1
    else:
        return 0

# Funcion que detecta los gestos
def detectHandGesture(numberOfFingers, fingerConvexityDefects, handRect):
    if handRect[2] > 25 and handRect[3] > 25:
        if numberOfFingers == 0:
            return 'Piedra'
        if numberOfFingers == 2:
            firstFinger, secondFinger, midPoint = fingerConvexityDefects[0]
            print(angle(firstFinger, secondFinger, midPoint))
            if angle(firstFinger, secondFinger, midPoint) > 70:
                return 'Pistola'
            else:
                return 'Peace'
    else:
        return ''

cap = cv2.VideoCapture(0) # 0 para que use la webcam
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

if not cap.isOpened: # En el caso de que no se pueda abrir la webcam, finaliza el programa
    print ("Unable to open cam")
    exit(0)
# Puntos para establecer el ROI
pt1 = (400,100)
pt2 = (600,300)

learningRates = (0.3,0) # Tupla de learning rates a emplear
currentLearningRateIndex = 0 # El learning rate que se emplea actualmente de la tupla

while True:
    ret, frame = cap.read()
    if not ret:
        exit(0)
    try:
        frame = cv2.flip(frame,1)
        roi = frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:] # Crea el ROI (zona a procesar)
        fgMask = backSub.apply(roi, None, learningRates[currentLearningRateIndex % 2])
        contours, hull = getcnthull(fgMask)
        # Se comprueba que al menos existe un contorno para seguir y que el learningRate esta bien establecido para que se bloquee el fondo actual
        if len(contours) > 0 and currentLearningRateIndex % 2 != 0:
            cv2.drawContours(frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:], contours, -1, (255,255,0), 2)
            cv2.drawContours(frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:], [hull], -1, (0, 255, 255), 2)
        else:
            contours = None
        if contours is not None:
            defects = getdefects(contours)
        else:
            defects = None
        if defects is not None:
            count = 0
            fingerConvexityDefects = list()
            # Por cada defecto de convexidad hallamos sus puntos start, end, far, su angulo y su profundidad
            for i in range(len(defects)):
                s,e,f,d = defects[i,0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                depth = d/256.0
                ang = angle(start,end,far)
                if depth > 30.0 and ang < 90:  # si se cumple, lo trata como un dedo a contar
                    cv2.circle(roi, far, 4, [0, 0, 255], -1)
                    fingerConvexityDefects.append((start,end,far))
            rect = cv2.boundingRect(contours)
            pt1a = (rect[0],rect[1])
            pt2a = (rect[0]+rect[2],rect[1]+rect[3])
            cv2.rectangle(frame[pt1[1]:pt2[1],pt1[0]:pt2[0],:],pt1a,pt2a,(0,0,255),3)
            count = countFingers(fingerConvexityDefects, rect)
            gesture = detectHandGesture(count,fingerConvexityDefects, rect)
            # Motrar que detectamos (numero de dedos y gesto)
            cv2.putText(frame, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
            cv2.putText(frame, gesture, (0, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        # Mostramos los distintos elementos es pantalla
        cv2.rectangle(frame,pt1,pt2,(255,0,0))
        cv2.imshow("frame", frame)
        cv2.imshow('ROI',roi)
        cv2.imshow('Foreground Mask',fgMask)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('s'): # s para cambiar entre acostumbrar al fondo o bloquearlo
        currentLearningRateIndex += 1
    if cv2.waitKey(1) & 0xFF == ord('q'): # q para quitar el programa
        break
cap.release()
cv2.destroyAllWindows()
