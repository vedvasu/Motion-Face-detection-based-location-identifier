import cv2.cv as cv
import face_detection_module as face        # import face detection module developed in openCV
import numpy as np
import cv2

class MotionDetectorInstantaneous():

    '''
    *Function Name: representation
    *Function Description: 
    - Used for representing output on the frame as well as on the console.
    - Faces are detected and value of distance is shown as pe the assumptions.
    *Assumptions: 
    - The dimmenssions of the room is 1066.8 cm x 609.6 (i.e 35x20 feet)
    - The distacne is shown form the opposite wall.
    '''

    def representation(self,r,curframe_face):

        x0,y0,x1,y1 = r
        cv2.rectangle(curframe_face, (x0,y0),(x1,y1),(0,255,0),1)
        c = (x1-x0)*(y1-y0)
        area_change_per_cm = (3600-729)/500     # As per the assumptions area of contour is 729 at center and 3600 at end
        distance = c/2

        cv2.putText(curframe_face, str(distance), (x0,y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
        print distance

    def onChange(self, val):        #callback when the user change the detection threshold
        self.threshold = val

    def __init__(self,threshold=1,showWindows=True):

        self.show = showWindows             #Either or not show the 2 windows
        self.frame = None
    
        self.capture=cv.CaptureFromFile('sample_vvs.3gp')
        self.frame = cv.QueryFrame(self.capture)            #Take a frame to init recorder
        
        self.frame1gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t-1
        cv.CvtColor(self.frame, self.frame1gray, cv.CV_RGB2GRAY)
        
        #Will hold the thresholded result
        self.res = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U)
        
        self.frame2gray = cv.CreateMat(self.frame.height, self.frame.width, cv.CV_8U) #Gray frame at t
        
        self.width = self.frame.width
        self.height = self.frame.height
        self.nb_pixels = self.width * self.height
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0                                           #Hold timestamp of the last detection
        #self.out = cv2.VideoWriter('output_vvs.avi', -1,25, (700,480))

    def run(self):
        #started = time.time()
        while True:
            
            curframe = cv.QueryFrame(self.capture)
            face_detector = face.CascadedDetector(cascade_fn="D:\e-yantra\opencv\sources\data\haarcascades\haarcascade_frontalface_alt2.xml")
            curframe_face = np.asarray(curframe[:,:])
            curframe_face = np.array(curframe_face, dtype=np.uint8)
            curframe_face = cv2.resize(curframe_face,(700,480))
    
            for i,r in enumerate(face_detector.detect(curframe_face)):
                self.representation(r,curframe_face)
            
            self.processImage(curframe)                 #Process the image
            
            if self.show:
                cv2.imshow('Image_face',curframe_face)
                #self.out.write(curframe_face)                          # to write uncomment line 56,75
                cv.ShowImage("Res", self.res)
                
            cv.Copy(self.frame2gray, self.frame1gray)
            c=cv.WaitKey(1) % 0x100
            if c==27 or c == 10:                            #Break if user enters 'Esc'.
                break            
    
    def processImage(self, frame):
        cv.CvtColor(frame, self.frame2gray, cv.CV_RGB2GRAY)
        
        #Absdiff to get the difference between to the frames
        cv.AbsDiff(self.frame1gray, self.frame2gray, self.res)
        
        #Remove the noise and do the threshold
        cv.Smooth(self.res, self.res, cv.CV_BLUR, 9,9)
        cv.Threshold(self.res, self.res, 10, 255, cv.CV_THRESH_BINARY_INV)

if __name__=="__main__":
    detect = MotionDetectorInstantaneous()
    detect.run()
