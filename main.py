import numpy as np
import cv2
import math

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

# Obtiene los pixeles que son piel
def skinmask(image): 
    hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8") # Menor rango de tono de piel
    upper = np.array([20, 255, 255], dtype = "uint8") # Mayor rango de tono de piel
    skin_region = cv2.inRange(hsvim, lower, upper) # Detectar pixeles en el rango
    blurred = cv2.blur(skin_region, (2,2))
    ret, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY) # Aplica lo del fondo negro
    return thresh

# Obtiene el contorno de la mano
def getcnthull(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(contours)
    return contours, hull

# Obtener defectos de convexidad
def getdefects(contours):
    hull = cv2.convexHull(contours, returnPoints=False)
    defects = cv2.convexityDefects(contours, hull) # Almacena los defectos de convexidad
    return defects

cap = cv2.VideoCapture(0) # '0' for webcam
while cap.isOpened():
    _, frame = cap.read()
    try:
        mask_img = skinmask(frame)
        contours, hull = getcnthull(mask_img)
        cv2.drawContours(frame, [contours], -1, (255,255,0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)
        defects = getdefects(contours)
        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i,0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                # a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                # b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                # c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                # angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
                ang = angle(start,end,far)
                # habría que añadir una condición para que haya distancia entre start y end
                if ang < 90:  # si el angulo es menor que 90
                    count += 1
                    cv2.circle(frame, far, 4, [0, 0, 255], -1)
            if count > 0:
                count = count+1
            cv2.putText(frame, str(count), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
        cv2.imshow("frame", frame)
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()