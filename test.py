import cv2
img = cv2.imread('test6.jpg')
print(img.shape)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print(gray.shape)
