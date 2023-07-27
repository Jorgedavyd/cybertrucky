import cv2
#Capturing the dataset
def main():
    
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224) # insert the width at which the model was trained 

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)# insert the height at which the model was trained (usually same as width) 

    cap.set(cv2.CAP_PROP_FPS, 30) # You can vary the frames based on the dataset len you want to achieve

    while (True):

        success, frame = cap.read()
        
        
        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


