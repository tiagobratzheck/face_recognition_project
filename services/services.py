"""

"""

import cv2
import numpy as np
from matplotlib import pyplot
import face_recognition
import dlib
import cmake
import os
import glob

from imutils.object_detection import non_max_suppression
import imutils

from datetime import datetime


class Service(object):


    def open_image(self, file):
        """[Return None]
            Read the image given.

        """
        image = cv2.imread(file)
        cv2.imshow("Image: ", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def cut_image(self, file, xh, xw, yh, yw):
        """[Return None]
            Cut the image given.

        """
        date = datetime.now()
        image = cv2.imread(file)
        cut = image[xh:xw, yh:yw]

        cv2.imwrite('imagens/imagens_salvas/cut_image_' +
                    date.strftime("%H-%M-%S") + '.jpg', cut)
        cv2.imshow("Recorte", cut)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def resize_image(self, file, width, height):
        """[Return None]
            Resize the image given

        """
        date = datetime.now()
        image = cv2.imread(file)

        r_image = cv2.resize(image, (width, height))
        cv2.imwrite('imagens/imagens_salvas/resized_image_' +
                    date.strftime("%H-%M-%S") + '.jpg', r_image)
        cv2.imshow("Image: ", r_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def color_image(self, file, color_option):
        """[Return None]
            Change the color of the image given

        color_option 1 = gray
        color_option 2 = hsv
        color_option 3 = lab

        """
        date = datetime.now()
        image = cv2.imread(file)

        if color_option == 1:
            imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('imagens/imagens_salvas/gray_image_' +
                        date.strftime("%H-%M-%S") + '.jpg', imageGray)
            cv2.imshow("Imagem Tons de Cinza", imageGray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif color_option == 2:
            imageHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imwrite('imagens/imagens_salvas/hsv_image_' +
                        date.strftime("%H-%M-%S") + '.jpg', imageHsv)
            cv2.imshow("Imagem HSV", imageHsv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            imageLab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            cv2.imwrite('imagens/imagens_salvas/lab_image_' +
                        date.strftime("%H-%M-%S") + '.jpg', imageLab)
            cv2.imshow("Imagem Lab", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def detect_person(self, file=None, algorithm=None):
        """[summary]

        """
        # if file:
        #    print("arquivo: " + file)
        # print("opção: " + str(algorithm))

        date = datetime.now()
        
        if algorithm == 1:

            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            if file == None:               

                cam = cv2.VideoCapture(0)
                while True:
                    retval, image = cam.read()
                    image = imutils.resize(image, width=min(600, image.shape[1]))
                    orig = image.copy()
                    (rects, weights) = hog.detectMultiScale(
                        image, winStride=(4, 4), padding=(8, 8), scale=1.05)
                    for (x, y, w, h) in rects:
                        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)                   
                    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                    for (xA, yA, xB, yB) in pick:
                        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    print("[INFO]: {} original boxes, {} after suppression".format(
                        len(rects), len(pick)))                  
                    cv2.imshow("After NMS", image)
                    k = cv2.waitKey(1) 
                    if k == 27:  
                        break
                cv2.destroyAllWindows()

            else:
                
                image = cv2.imread(file)
                while True:                    
                    image = imutils.resize(image, width=min(600, image.shape[1]))
                    orig = image.copy()
                    (rects, weights) = hog.detectMultiScale(
                        image, winStride=(4, 4), padding=(8, 8), scale=1.05)
                    for (x, y, w, h) in rects:
                        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)                  
                    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
                    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
                    for (xA, yA, xB, yB) in pick:
                        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                    print("[INFO]: {} original boxes, {} after suppression".format(
                        len(rects), len(pick)))                   
                    cv2.imshow("After NMS", image)
                    k = cv2.waitKey(1)  
                    if k == 27:  
                        break
                cv2.destroyAllWindows()

        else:      
                  
            faces_encodings = []
            faces_names = []
            currentDir = os.getcwd()
            path = os.path.join(currentDir, "imagens/condominos/")

            lista = [f for f in glob.glob(path+"*.jpg")]            
            tamanhoLista = len(lista)
            names = lista.copy()

            for i in range(tamanhoLista):
                presets = face_recognition.load_image_file(lista[i])
                encoding = face_recognition.face_encodings(presets)[0]
                faces_encodings.append(encoding)
                names[i] = names[i].replace(currentDir, "")
                names[i] = names[i].replace(".jpg", "")
                names[i] = names[i].replace("imagens", "")
                names[i] = names[i].replace("condominos", "")
                names[i] = names[i].replace("/", "")
                names[i] = names[i].replace("\\", "")
                faces_names.append(names[i])

            face_locations = []
            face_encodings = []
            face_names = []
            
            if file == None:

                camera = cv2.VideoCapture(0)
                while True:

                    _, frame = camera.read()
                    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgbSmallFrame = smallFrame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgbSmallFrame)
                    face_encodings = face_recognition.face_encodings(
                        rgbSmallFrame, face_locations)
                    face_names = []
                    for face in face_encodings:
                        matches = face_recognition.compare_faces(faces_encodings, face)
                        name = "Desconhecida"
                        face_distances = face_recognition.face_distance(faces_encodings, face)
                        bestMatch = np.argmin(face_distances)
                        if matches[bestMatch]:
                            name = faces_names[bestMatch]
                        face_names.append(name)

                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top = top * 4
                        right = right * 4
                        bottom = bottom * 4
                        left = left * 4
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
                        cv2.putText(frame, name, (left+6, bottom-6),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                        cv2.imwrite('imagens/imagens_de_log/pessoa_'+ name +'_'+ date.strftime("%H-%M-%S") +'.jpg', frame)
                    cv2.imshow("Camera", frame)   
                    #cv2.imwrite('imagens/imagens_de_log/morador_'+ date.strftime("%H-%M-%S") +'.jpg', frame)                 
                    k = cv2.waitKey(30)
                    if k == 27:
                        break
                cv2.destroyAllWindows()
                camera.release()
            
            else:
            
                image = cv2.imread(file)  
                while True:
                    
                    smallFrame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                    rgbSmallFrame = smallFrame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgbSmallFrame)
                    face_encodings = face_recognition.face_encodings(
                        rgbSmallFrame, face_locations)
                    face_names = []
                    for face in face_encodings:
                        matches = face_recognition.compare_faces(faces_encodings, face)
                        name = "Desconhecida"
                        face_distances = face_recognition.face_distance(faces_encodings, face)
                        bestMatch = np.argmin(face_distances)
                        if matches[bestMatch]:
                            name = faces_names[bestMatch]
                        face_names.append(name)

                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top = top * 4
                        right = right * 4
                        bottom = bottom * 4
                        left = left * 4
                        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.rectangle(image, (left, bottom-35), (right, bottom), (0,0,255), cv2.FILLED)
                        cv2.putText(image, name, (left+6, bottom-6),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                        cv2.imwrite('imagens/imagens_de_log/morador_'+ name +'_'+ date.strftime("%H-%M-%S") +'.jpg', image)
                    cv2.imshow("Imagem", image)
                    #cv2.imwrite('imagens/imagens_de_log/morador_'+ date.strftime("%H-%M-%S") +'.jpg', image)
                    k = cv2.waitKey(30)
                    if k == 27:
                        break
                cv2.destroyAllWindows()


    def get_object(self, file, color):
        """[summary]

        color:
            1 - PINK 
            2 - AMARELO
            3 - VERDE NEON
        """
        if color == 1:
            lower = np.array([147, 20, 255])            
        elif color == 2:
            lower = np.array([0, 255, 255])
        else:
            lower = np.array([0, 252, 124])

        date = datetime.now()
            
        if file == None:

            camera = cv2.VideoCapture(0) 
            while True:
                _, frame = camera.read() 
                frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                upper = np.array([255, 255, 255]) 
                mascara = cv2.inRange(frameHsv, lower, upper) 
                resultado = cv2.bitwise_and(frame, frame, mask=mascara) 
                frameGray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY) 
                _, thresh = cv2.threshold(frameGray, 3, 255, cv2.THRESH_BINARY) 
                contornos, _ = cv2.findContours(
                    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                         
                for contorno in contornos: 
                    (x, y, w, h) = cv2.boundingRect(contorno) 
                    area = cv2.contourArea(contorno) 
                    if area > 1500:  
                        cv2.putText(frame, "Objeto detectado", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255)) 
                        cv2.drawContours(frame, contorno, -1, (0, 0, 0), 5) 
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1) 
                        cv2.putText(frame, "X: " + str(x) + "Y: " + str(y),
                                    (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255)) 
                    cv2.imwrite('imagens/imagens_de_log/objeto_detectado_' + date.strftime("%H-%M-%S") +'.jpg', frame)            
                cv2.imshow("Camera", frame) 
                key = cv2.waitKey(60) 
                if key == 27:   
                    break
            cv2.destroyAllWindows() 
            camera.release() 

        else:

            image = cv2.imread(file)
            while True:                
                frameHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                upper = np.array([255, 255, 255]) 
                mascara = cv2.inRange(frameHsv, lower, upper) 
                resultado = cv2.bitwise_and(frame, frame, mask=mascara) 
                frameGray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY) 
                _, thresh = cv2.threshold(frameGray, 3, 255, cv2.THRESH_BINARY) 
                contornos, _ = cv2.findContours(
                    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)                         
                for contorno in contornos: 
                    (x, y, w, h) = cv2.boundingRect(contorno) 
                    area = cv2.contourArea(contorno) 
                    if area > 1500: 
                        cv2.putText(frame, "Objeto detectado", (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255)) 
                        cv2.drawContours(frame, contorno, -1, (0, 0, 0), 5) 
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1) 
                        cv2.putText(frame, "X: " + str(x) + "Y: " + str(y),
                                    (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))
                    cv2.imwrite('imagens/imagens_salvas/imagens_de_log/objeto_detectado_' + date.strftime("%H-%M-%S") +'.jpg', frame)              
                cv2.imshow("Imagem", frame) 
                key = cv2.waitKey(60) 
                if key == 27:   
                    break
            cv2.destroyAllWindows()              

    
    def binarization(self, file):
        """[summary]

        """
        date = datetime.now()
        image = cv2.imread(file)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        blur = cv2.GaussianBlur(gray, (7, 7), 0)        
        _, imageThrehold = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)  
        cv2.imwrite('imagens/imagens_salvas/binary_image_' + date.strftime("%H-%M-%S") +'.jpg', imageThrehold)        
        cv2.imshow("Threhold", imageThrehold)        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def morphology(self, file, n_iter):
        """[summary]

        """
        date = datetime.now()
        image = cv2.imread(file, 0)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(image, kernel, iterations = n_iter)

        dilation = cv2.dilate(image, kernel, iterations = n_iter)
        cv2.imwrite('imagens/imagens_salvas/imagens_morfo/dilation_image_' + date.strftime("%H-%M-%S") +'.jpg', dilation)

        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite('imagens/imagens_salvas/imagens_morfo/opening_image_' + date.strftime("%H-%M-%S") +'.jpg', opening)

        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('imagens/imagens_salvas/imagens_morfo/closing_image_' + date.strftime("%H-%M-%S") +'.jpg', closing)

        gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite('imagens/imagens_salvas/imagens_morfo/gradient_image_' + date.strftime("%H-%M-%S") +'.jpg', gradient)

        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        cv2.imwrite('imagens/imagens_salvas/imagens_morfo/tophat_image_' + date.strftime("%H-%M-%S") +'.jpg', tophat)

        blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite('imagens/imagens_salvas/imagens_morfo/blackhat_image_' + date.strftime("%H-%M-%S") +'.jpg', blackhat)



    def hsv(self, p_file, option):
        """[summary]

        """
        def skip():
            pass

        if option == 4:
            cv2.namedWindow("FRAME")
            cv2.createTrackbar("h", 'FRAME', 0, 179, skip)
            cv2.createTrackbar("s", 'FRAME', 0, 255, skip)
            cv2.createTrackbar("v", 'FRAME', 0, 255, skip)

            camera = cv2.VideoCapture(0)
            while True:
                _, frame = camera.read()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h = cv2.getTrackbarPos("h", "FRAME")
                s = cv2.getTrackbarPos("s", "FRAME")
                v = cv2.getTrackbarPos("v", "FRAME")
                lower = np.array([h, s, v])
                upper = np.array([255, 255, 255])
                mascara = cv2.inRange(hsv, lower, upper)
                resultado = cv2.bitwise_and(frame, frame, mask=mascara)
                cv2.imshow("FRAME", resultado)
                key = cv2.waitKey(60)
                if key == 27:
                    break
            cv2.destroyAllWindows()
            camera.release()
        else:
            cv2.namedWindow("FRAME")
            cv2.createTrackbar("h", 'FRAME', 0, 179, skip)
            cv2.createTrackbar("s", 'FRAME', 0, 255, skip)
            cv2.createTrackbar("v", 'FRAME', 0, 255, skip)

            image = cv2.imread(p_file)
            while True:                
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                h = cv2.getTrackbarPos("h", "FRAME")
                s = cv2.getTrackbarPos("s", "FRAME")
                v = cv2.getTrackbarPos("v", "FRAME")
                lower = np.array([h, s, v])
                upper = np.array([255, 255, 255])
                mascara = cv2.inRange(hsv, lower, upper)
                resultado = cv2.bitwise_and(image, image, mask=mascara)
                cv2.imshow("FRAME", resultado)
                key = cv2.waitKey(60)
                if key == 27:
                    break
            cv2.destroyAllWindows()
            
        
