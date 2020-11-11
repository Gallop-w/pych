import cv2
# 读取图片
img = 'orange.jpg'
img = cv2.imread(img)
sp = img.shape
print(sp[0],sp[1])
im2 = cv2.resize(img,(299,299),)  # 为图片重新指定尺寸
cv2.imwrite('orange.jpg',im2)
sp = im2.shape
print(sp[0],sp[1])

# # 选择ROI
# roi = cv2.selectROI(windowName="original", img=img, showCrosshair=True, fromCenter=False)
# x, y, w, h = roi
# print(roi)
# 
# # 显示ROI并保存图片
# if roi != (0, 0, 0, 0):
#     crop = img[y:y+h, x:x+w]
#     cv2.imshow('crop', crop)
#     cv2.imwrite('crop.jpg', crop)
#     print('Saved!')
# 
# # 退出
cv2.waitKey(0)
cv2.destroyAllWindows()