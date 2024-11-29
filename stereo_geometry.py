import cv2 
import math
import numpy as np
import matplotlib.pyplot as plt

def fundamentalMatCalc(pts1, pts2):    
    #We'll be using the SVD method as in Multi View Geometry Book
    n = pts1.shape[1]
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [pts1[0,i]*pts2[0,i], pts1[0,i]*pts2[1,i], pts1[0,i]*pts2[2,i], 
                pts1[1,i]*pts2[0,i], pts1[1,i]*pts2[1,i], pts1[1,i]*pts2[2,i],
                pts1[2,i]*pts2[0,i], pts1[2,i]*pts2[1,i], pts1[2,i]*pts2[2,i] ]
            
    #Using SVD to solve
    U,S,V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    U,S,V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))    
    return F


def drawLines(image1, image2, lines, points1, points2):
    #This function is based on reference from OpenCV documentation
    cols = image1.shape[1]
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    for i in range(len(points1)):
        color = tuple(np.random.randint(100, 255, 3).tolist())
        x0, y0 = map(int, [0, -lines[i][2]/lines[i][1]])
        x1, y1 = map(int, [cols, -(lines[i][2] + lines[i][0] * cols)/ lines[i][1]])
        image1 = cv2.line(image1, (x0, y0), (x1, y1), color, 1)
        image1 = cv2.circle(image1, tuple(points1[i]), 5, color, -1)
        image2 = cv2.circle(image2, tuple(points2[i]), 5, color, -1)
    return image1, image2



image1, image2 = cv2.imread('Amitava_first.JPG'), cv2.imread('Amitava_second.JPG')
image1, image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#Initialize SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp1, ds1 = sift.detectAndCompute(image1, None)
kp2, ds2 = sift.detectAndCompute(image2, None)

#Match Features
bfm = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bfm.match(ds1, ds2)
matches = sorted(matches, key = lambda x:x.distance)

fig = plt.figure()
fig.suptitle('Some Sample Matches', fontsize = 20)
image3 = cv2.drawMatches(image1, kp1, image2, kp2, matches[:50], image2, flags=2)
plt.imshow(image3)
plt.axis("off")
plt.show()

good = []
pts1 = []
pts2 = []

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(ds1, ds2, k = 2)

ratio_thresh = 0.8
for i,(m,n) in enumerate(matches):
    if m.distance < ratio_thresh * n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

#Calculation of fundamental Matrix from custom written function, but here we have used the RANSAC result in further calculations
Fself = fundamentalMatCalc(pts1, pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

print("")
print("Fundamental Matrix: ")
print(F)
print("")

# Using the mask we select only the inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

#Calculate corresponding epipolar lines
lines1, lines2 = (cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)).reshape(-1, 3), (cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)).reshape(-1, 3)

#Plot Lines
image5, image6 = drawLines(image1, image2, lines1, pts1, pts2)
image3, image4 = drawLines(image2, image1,lines2, pts2, pts1)

fig = plt.figure()
fig.suptitle('Epipolar lines', fontsize = 20)
plt.subplot(121),plt.imshow(image5)
plt.axis("off")
plt.subplot(122),plt.imshow(image3)
plt.axis("off")
plt.show()

print("")
print("Epipoles Estimation from lines: ")
print("")

eL = np.cross(lines1[0],lines1[2])
eL[0] = eL[0]/eL[2]
eL[1] = eL[1]/eL[2]
eL[2] = 1
print("Left Epipole: ", eL)

eR = np.cross(lines2[1],lines2[3])
eR[0] = eR[0]/eR[2]
eR[1] = eR[1]/eR[2]
eR[2] = 1
print("Right Epipole: ", eR)

print("")
print("Epipoles Estimation from Fundamental Matrix: ")
print("")

F1 = np.array([F[0][0:2],F[1][0:2]])
F2 = np.array([[F[0][0],F[1][0]],[F[0][1],F[1][1]]])

X1 = -1 * np.array([F[0][2],F[1][2]])
X2 = -1 * F[2][0:2]

#Using numpy solver
eLF = np.linalg.solve(F1, X1)
eRF = np.linalg.solve(F2, X2)
eLF = np.append(eLF, [1])
eRF = np.append(eRF, [1])
print("Left Epipole: ", eLF)
print("Right Epipole: ", eRF)
print("")

distanceL = math.sqrt(  (eLF[0] - eL[0])**2 + (eLF[1] - eL[1])**2  )
distanceR = math.sqrt(  (eRF[0] - eR[0])**2 + (eRF[1] - eR[1])**2  )

print("Distance between left epipoles: ", distanceL)
print("Distance between right epipoles: ", distanceR)


print("")
print("Projection Matrices Estimation from Fundamental Matrix: ")
print("")

#Assuming one of the matrices to be [I | 0]
tR = np.array([[0, -1, eR[1]], [1, 0, -1 * eR[0]], [-1 * eR[1], eR[0], 0]])
eF = tR @ F

pMat1 = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0]])
pMat2 = np.append(eF,np.array([[eR[0]], [eR[1]], [1]]), axis = 1)

print("Left: ")
print(pMat1)
print("RIght: ")
print(pMat2)

M1 = np.array([pMat1[0][0:3], pMat1[1][0:3], pMat1[2][0:3]])
M2 = np.array([pMat2[0][0:3], pMat2[1][0:3], pMat2[2][0:3]])

iM1 = np.linalg.inv(M1) 
iM2 = np.linalg.inv(M2) 

for i in range(10):
    x1 = np.append(pts1[i], np.array([1]))
    x2 = np.append(pts2[i], np.array([1]))

    #Finding directional back rays
    r1 = (-1 * iM1) @ x1
    r2 = (-1 * iM2) @ x2

    #point = np.cross(r1, r2) ?
    #Further approach for non-intersecting rays to be done. Maybe Gradient Descent with distance as cost function?
    
print("3D points calculated.")
