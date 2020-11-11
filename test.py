import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Load an color image in grayscale
img = cv2.imread('testimg.jpg',0)
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)
# cv2.waitKey(0)&0xFF
# cv2.destroyAllWindows()
# cv2.imwrite('messigray.png',img)

# k = cv2.waitKey(0)&0xFF
# if k == 27: # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray1.png',img)
#     cv2.destroyAllWindows()

# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([]) # to hide tick values on X and Y axis
# plt.show()


# # Create a black image
# img=np.zeros((512,512,3), np.uint8)
#  # Draw a diagonal blue line with thickness of 5 px
# cv2.line(img,(0,0),(511,511),(255,0,0),5)
# # Draw a rectangle,point and color
# cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
# # Draw a circle
# cv2.circle(img,(447,63), 63, (0,0,255), -1)
# # Draw a ellipse,圆心，长短轴，旋转角，圆弧旋转角，色，线
# cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
# pts=np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts=pts.reshape((-1,1,2))
# # 这里 reshape 的第一个参数为-1, 表明这一维的长度是根据后面的维度的计算出来的。
# font=cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)
#
# winname = 'example'
# cv2.namedWindow(winname)
# cv2.imshow(winname, img)
# cv2.waitKey(0)
# cv2.destroyWindow(winname)

# import cv2
# events=[i for i in dir(cv2) if 'EVENT'in i]
# print (events)
#
# import cv2
# import numpy as np
# #mouse callback function
# def draw_circle(event,x,y,flags,param):
#     if event==cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#
# # 创建图像与窗口并将窗口与回调函数绑定
# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
# # 无限等待，直到输入esc
# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20)&0xFF==27:
#         break
# cv2.destroyAllWindows()


# drawing = False
# mode = True
# ix, iy = -1, -1
#
# def draw_circle(event, x, y, flags, param):
#     global ix, iy, drawing, mode
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix, iy =x, y
# # 当鼠标左键按下并移动是绘制图形。event 可以查看移动，flag 查看是否按下
#     elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
#         if drawing==True:
#             if mode==True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),3)
#             else:
#                 # 绘制圆圈，小圆点连在一起就成了线，3 代表了笔画的粗细
#                 cv2.circle(img,(x,y),3,(0,0,255),-1) # 下面注释掉的代码是起始点为圆心，起点到终点为半径的
#                 # r=int(np.sqrt((x-ix)**2+(y-iy)**2))
#                 # cv2.circle(img,(x,y),r,(0,0,255),-1)
# # 当鼠标松开停止绘画。
#     elif event==cv2.EVENT_LBUTTONUP:
#         drawing==False
#         if mode==True:
#             cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         else:
#             cv2.circle(img,(x,y),5,(0,0,255),-1)
#
# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
# while(1):
#     cv2.imshow('image',img)
#     k=cv2.waitKey(1)&0xFF
#     if k==ord('m'):
#         mode=not mode
#     elif k==27:
#         break


# 回调函数
# def nothing(x):
#     pass
#
# # 创建黑色图像,长、宽、通道数
# img = np.zeros((300,512,3), np.uint8)
# cv2.namedWindow('image')
#
# cv2.createTrackbar('R', 'image', 0, 255, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('B', 'image', 0, 255, nothing)
#
# switch = '0:OFF\n1:ON'
# cv2.createTrackbar(switch, 'image', 0, 1, nothing)
#
# while(1):
#     cv2.imshow('image', img)
#     k = cv2.waitKey(1)&0xFF
#     if k == 27:
#         break
#
#     r = cv2.getTrackbarPos('R', 'image')
#     g = cv2.getTrackbarPos('G', 'image')
#     b = cv2.getTrackbarPos('B', 'image')
#     s = cv2.getTrackbarPos(switch, 'image')
#
#     if s == 0:
#         img[:] = 0
#     else:
#         img[::] = [b, g, r]
#
# cv2.destroyWindow()

img = cv2.imread('testimg.jpg')
# px=img[100,100]
# print (px)
# blue=img[100,100,0]
# print (blue)
# a = img.item(10,10,2)
# print(a)
# img.itemset((10,10,2),101)
# print (img.item(10,10,2))
# print (img.size)
# print (img.shape)
# print (img.dtype)
# ball=img[0:0,500:500]
# img[1000:1000,1500:1500]=ball
# b=img[:,:,0]
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20)&0xFF==27:
#         break
# cv2.destroyAllWindows()
# replicate = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_CONSTANT,value=180)
#
# plt.subplot(231),plt.imshow(img,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
# plt.show()

# x = np.uint8([250])
# y = np.uint8([10])
# print (cv2.add(x,y)) # 250+10 = 260 => 255
# print (x+y) # 250+10 = 260 % 256 = 4
import time
# tic = time.time()
# img1 = cv2.imread('test1.jpg')
# img2 = cv2.imread('test2.jpg')
# img1=np.zeros((512,512,3),np.uint8)
# img2=np.zeros((512,512,3),np.uint8)
# img1[::] = [25,180,65]
# img2[::] = [125,80,165]
# # dst=cv2.addWeighted(img1,0.3,img2,0.7,0)
# # cv2.imshow('dst',dst)
# # cv2.waitKey(0)
# # cv2.destroyAllWindow()
#
# toc = time.time()
# a= toc - tic
# print(a)

# set blue thresh
# lower_yellow = np.array([11, 43, 46])
# upper_yellow = np.array([25, 255, 255])
#
# frame = cv2.imread("test1.jpg")  # 读取图像
# cv2.imshow("who", frame)
#
# # compress
# cp = cv2.resize(frame, (300, 300), interpolation=cv2.INTER_AREA) # interpolation 指定重新计算像素的方式
# cv2.imwrite("tulips_1.jpg", cp)
#
# # change to hsv model
# hsv = cv2.cvtColor(cp, cv2.COLOR_BGR2HSV)
#
# # get mask
# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
# cv2.imshow('Mask', mask)
#
# # detect blue
# res = cv2.bitwise_and(cp, cp, mask=mask)
# cv2.imshow('Result', res)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img = cv2.imread("test1.jpg",0)
# qq = cv2.resize(frame, None, fx = 0.1, fy = 0.1, interpolation=cv2.INTER_AREA)
# while(1):
#     cv2.imshow('res',qq)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

# 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
# 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
# cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
# cv2.imshow('img',0)
# rows,cols=img.shape
# print(rows,cols)
# M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.7)
# # 第三个参数是输出图像的尺寸中心
# dst=cv2.warpAffine(img,M,(2*cols,2*rows))
# while(1):
#     #cv2.imshow('img',dst)
#     #cv2.imshow('img1', M)
#     cv2.imshow('img2', img)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cv2.destroyAllWindows()


# img=cv2.imread('test1.jpg',0)
# # 第四个参数为不同的阈值方法，两个返回值
# ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
# ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
# ret,thresh5=cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()

# # global thresholding
# ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# # Otsu's thresholding
# ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
# #（5,5）为高斯核的大小，0 为标准差
# blur = cv2.GaussianBlur(img,(5,5),0)
# # 阈值一定要设为 0！
# ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# # plot all the images and their histograms
# images =    [img, 0, th1,
#             img, 0, th2,
#             blur, 0, th3]
# titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
# 'Original Noisy Image','Histogram',"Otsu's Thresholding",
# 'Gaussian filtered Image','Histogram',"Otsu's Thresholding"] # 这里使用了 pyplot 中画直方图的方法，plt.hist, 要注意的是它的参数是一维数组
# # 所以这里使用了（numpy）ravel 方法，将多维数组转换成一维，也可以使用 flatten 方法
# #ndarray.flat 1-D iterator over an array.
# #ndarray.flatten 1-D array copy of the elements of an array in row-major order.
# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()

kernel = np.ones((5,5),np.float32)/25
#cv.Filter2D(src, dst, kernel, anchor=(-1, -1))
#ddepth –desired depth of the destination image;
#if it is negative, it will be the same as src.depth();
#the following combinations of src.depth() and ddepth are supported:
#src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
#src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
#src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
#src.depth() = CV_64F, ddepth = -1/CV_64F
#when ddepth=-1, the output image will have the same depth as the source.
# dst = cv2.filter2D(img,-1,kernel)
# blur = cv2.bilateralFilter(img,9,75,75)
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(blur),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()

img = cv2.imread('test3.jpg',0)
# kernel = np.ones((5,5),np.uint8)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
# while(1):
#     cv2.imshow('img',tophat)
#     if cv2.waitKey(1)&0xFF==27:
#         break
# cv2.destroyAllWindows()

# edges = cv2.Canny(img,100,200)
# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# img = cv2.imread('test3.jpg')
# lower_reso = cv2.pyrDown(img)
# higher_reso = cv2.pyrUp(img)
# cv2.imshow('111',img)
# cv2.imshow('result1.jpg',lower_reso)
# cv2.imshow('result2.jpg',higher_reso)
# cv2.waitKey(0)

# A = cv2.imread('apple.jpg')
# B = cv2.imread('orange.jpg')
# sp1, sp2 = A.shape, B.shape
# print(sp1[0],sp1[1],sp2[0],sp2[1])
# # generate Gaussian pyramid for A
# G = A.copy()
# gpA = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpA.append(G)
# # generate Gaussian pyramid for B
# G = B.copy()
# gpB = [G]
# for i in range(6):
#     G = cv2.pyrDown(G)
#     gpB.append(G)
# generate Laplacian Pyramid for A
# lpA = [gpA[5]]
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpA[i])
#     L = cv2.subtract(gpA[i-1],GE)
#     lpA.append(L)
# # generate Laplacian Pyramid for B
# lpB = [gpB[5]]
# for i in range(5,0,-1):
#     GE = cv2.pyrUp(gpB[i])
#     L = cv2.subtract(gpB[i-1],GE)
#     lpB.append(L)
# # Now add left and right halves of images in each level
# #numpy.hstack(tup)
# #Take a sequence of arrays and stack them horizontally
# #to make a single array.
# LS = []
# for la,lb in zip(lpA,lpB):
#     rows,cols,dpt = la.shape
#     ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
#     LS.append(ls)
# # now reconstruct
# ls_ = LS[0]
# for i in range(1,6):
#     ls_ = cv2.pyrUp(ls_)
#     ls_ = cv2.add(ls_, LS[i])
# # image with direct connecting each half
# real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
# cv2.imwrite('Pyramid_blending2.jpg',ls_)
# cv2.imwrite('Direct_blending.jpg',real)


im = cv2.imread('light.JPG')
img = cv2.imread('light.JPG')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh= cv2.threshold(imgray,127,255,0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#img= cv2.drawContours(im, contours, -1, (0,255,0), 3)
# cv2.imshow('result.jpg',img)
# 提取轮廓
cnt = contours[0]
# x,y,w,h = cv2.boundingRect(cnt)
# img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
#
# print(cnt)
# print (len(cnt))

# (x,y),radius = cv2.minEnclosingCircle(cnt)
# center = (int(x),int(y))
# radius = int(radius)
# img = cv2.circle(img,center,radius,(0,255,0),2)

# ellipse = cv2.fitEllipse(cnt)
# img = cv2.ellipse(im,ellipse,(0,255,0),2)


# cv2.imshow('img2', img)
#
# x,y,w,h = cv2.boundingRect(cnt)
# aspect_ratio = float(w)/h
# area = cv2.contourArea(cnt)
# rect_area = w * h
# extent = float(area)/rect_area
# hull = cv2.convexHull(cnt)
# hull_area = cv2.contourArea(hull)
# solidity = float(area)/hull_area
# equi_diameter = np.sqrt(4*area/np.pi)

# print(aspect_ratio, extent, solidity, equi_diameter)

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
cv2.circle(img, leftmost, 5, (0, 0, 255), -1)
cv2.imshow('img2', img)

cv2.waitKey(0)
