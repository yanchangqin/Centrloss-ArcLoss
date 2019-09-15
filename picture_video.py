import cv2
import os
path = r'F:\ycq\centerloss\adm-sum'
filelist = os.listdir(path)
file = []
for i in filelist:
    i_ = i.split('.')[0]
    file.append(int(i_))
file.sort()
file_list = []
for j in file:
    file = os.path.join('{}\{}.jpg'.format(path,j))
    file_list.append(file)

fps =10
size = (640,480)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video = cv2.VideoWriter("VideoTest1.avi", fourcc, fps, size)
for item in file_list:
    img = cv2.imread(item)
    video.write(img)
video.release()
cv2.destroyAllWindows()
print('完成')

