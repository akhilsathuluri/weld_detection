import cv2

cam = cv2.VideoCapture(0)
cv2.namedWindow("Display")

template = cv2.imread('background.png')
# cv2.imshow('show', template)
# cv2.waitKey()
_, w, h = template.shape[::-1]
# print(template.shape[::-1])
method = eval('cv2.TM_CCOEFF')

img_counter = 0
while True:
    ret, frame = cam.read()
    cv2.imshow('Display', frame)

    res = cv2.matchTemplate(frame,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(max_val)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print('Escape hit. Closing...')
        break
    elif k%256 == 32:
        img_name = "image_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written".format(img_name))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()
