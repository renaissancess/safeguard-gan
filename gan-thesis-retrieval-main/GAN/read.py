import os
import cv2
import numpy as np

#### 将video中的图片合到一个文件夹中

# pa_path="C://Users//hp//Desktop//safe//video"

# dirs=[]
# for root,dir,files in os.walk(pa_path):
#     # root指当前路径，dirs指文件夹中所有文件名的列表,
#     dirs=dir
#     break

# i=0

# for dir in dirs:  #遍历12个视频截图文件夹
#     path=pa_path+"/"+dir+"/"+"pic"
#     for filename in os.listdir(path):
#         oldpath=path+"/"+filename
#         img=cv2.imread(oldpath)
#         newname="output"+str(i)+".png"
#         i=i+1
#         cv2.imwrite("C://Users//hp//Desktop//safe//data"+"//"+newname,img)

#### 将video中的label合到一个文件夹中

pa_path="C://Users//hp//Desktop//safe//video"

dirs=[]
for root,dir,files in os.walk(pa_path):
    # root指当前路径，dirs指文件夹中所有文件名的列表,
    dirs=dir
    break

i=0
import shutil

for dir in dirs:  #遍历12个视频截图文件夹
    path=pa_path+"/"+dir+"/"+"label"
    for filename in os.listdir(path):
        oldpath=path+"/"+filename
        newname="output"+str(i)+".txt"
        i=i+1
        shutil.move(oldpath,"C://Users//hp//Desktop//safe//label"+"//"+newname)
   
# # 读取图片
# path="C://Users//hp//Desktop//safe//data"

# img_col=np.array(())
# # opencv读取图片是np格式，
# for filename in os.listdir(path):
#     img= cv2.imread(path+"/"+filename)
#     np.append(img_col,img)

# print(img_col.size)
