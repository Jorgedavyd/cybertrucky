import cv2
import os
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive



#Setting up the camera

#Folder

classes = int(input('classes: '))
    

file_list = []


#Capturing the dataset
def capture(frameNr):
    
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224) # insert the width at which the model was trained 

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)# insert the height at which the model was trained (usually same as width) 

    cap.set(cv2.CAP_PROP_FPS, 30) # You can vary the frames based on the dataset len you want to achieve

    while (True):

        success, frame = cap.read()
        
        
        cv2.imshow('frame', frame)
        
        if success:
            ##include google drive
            file_list.append(frame)

        else:
            raise(RuntimeWarning)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameNr = frameNr+1

    cap.release()
    cv2.destroyAllWindows()
        

if __name__ == '__main__':
    capture()

input('After getting the dataset, you are able to delete this file')