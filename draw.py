import cv2
import numpy as np
from scipy.spatial import distance as dist
import utils
import imutils
from imutils import perspective
from imutils import contours

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
points = []
coinActualSize = 23
dA, dB = None, None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing, points

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y
        points.append((x,y))

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(orig,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)
            pt1_x,pt1_y=x,y
            points.append((x,y))
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(orig,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=2)
        total_dist = 0
        for i in range(1, len(points)):
            adj_distance = dist.euclidean(points[i-1], points[i] )
            total_dist += adj_distance
        print("Total distance is " + str(total_dist*coinActualSize/dA) + " mm")
        points = []
              

# img = np.zeros((512,512,3), np.uint8)
img = cv2.imread('image0.jpeg')
img = utils.resizeWithAspectRatio(img, width=600)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)

#drawing bounding box for coin
coint_cnt = cnts[0]
orig = img.copy()
box = cv2.minAreaRect(coint_cnt)
box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
box = np.array(box, dtype="int")

box = perspective.order_points(box)
cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
for (x, y) in box:
	cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

(tl, tr, br, bl) = box
(tltrX, tltrY) = utils.midpoint(tl, tr)
(blbrX, blbrY) = utils.midpoint(bl, br)

(tlblX, tlblY) = utils.midpoint(tl, bl)
(trbrX, trbrY) = utils.midpoint(tr, br)

# cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
# cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
# cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
# cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

# cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
# cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)


dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


# dimA = dA / pixelsPerMetric
# dimB = dB / pixelsPerMetric

# cv2.putText(orig, "{:.1f}mm".format(dimA),
# 		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
# 		0.65, (255, 255, 255), 2)
# cv2.putText(orig, "{:.1f}mm".format(dimB),
# 		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
# 		0.65, (255, 255, 255), 2)

cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',orig)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()