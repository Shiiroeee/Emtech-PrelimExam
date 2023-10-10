import cv2
import filters
from managers import WindowManager, CaptureManager

class Cameo (object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onkeypress)
        self._captureManager = CaptureManager( cv2.VideoCapture(0), self._windowManager, True)

        self._sharpenFilter = filters.SharpenFilter()
        self._findEdgesFilter = filters.FindEdgesFilter()
        self._blurFilter = filters.Blurfilter()
        self._embossFilter = filters.EmbossFilter()
        
    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
             self._captureManager.enterFrame()
             frame = self._captureManager.frame  
             #self._sharpenFilter.apply(frame,frame)
             #self._findEdgesFilter.apply(frame, frame)
             #self._blurFilter.apply(frame, frame)
             #self._embossFilter.apply(frame, frame)
             #filters.strokeEdges(frame, frame)
             filters.ContourFilter(frame)
             edges = cv2.Canny(frame, 50, 150)
             contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
             circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=60, minRadius=5, maxRadius=100)
            
             if circles is not None:
                circles = circles.astype(int)
                for circle in circles[0, :]:
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    cv2.circle(frame, center, radius, (0, 255, 0), 2)
                    cv2.circle(frame, center, 2, (0, 0, 255), 3)

             self._windowManager.show(frame)
             self._captureManager.exitFrame()
             self._windowManager.processEvents()


    # The rest should remain the same
    def onkeypress(self, keycode):
         if keycode == 32: 
            self._captureManager.writeImage('screenshot.png')
         elif keycode == 9: 
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo( 'screencast.avi')
            else:
                self._captureManager.stoptWritingVideo()
         elif keycode == 27: 
            self._windowManager.destroyWindow()
         
if __name__ == "__main__":
    Cameo().run()
 