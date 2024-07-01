import os
import glob
origin_path = os.getcwd()
os.chdir('/home/sam/Desktop/multiobject/rgbd')
print(os.listdir())
path='/home'
depth = glob.glob(os.path.join(path,'depth_*.png'))
depth.sort()
print(depth[0:10])

rgb=glob.glob(os.path.join(path,'rgb*.jpg'))
rgb.sort()
print(rgb[0:10])

label=glob.glob(os.path.join(path,'rgb_*.txt'))
label.sort()
print(label[0:10])

from PIL import Image
import matplotlib.pyplot as plt

def str2num(point):
    x, y = point.split()
    x, y = int(round(float(x))), int(round(float(y)))

    return (x, y)


def get_rectangle(cornell_grasp_file):
    grasp_rectangles = []
    with open(cornell_grasp_file, 'r') as f:
        while True:
            grasp_rectangle = []
            point0 = f.readline().strip()
            if not point0:
                break
            point1, point2, point3 = f.readline().strip(), f.readline().strip(), f.readline().strip()
            grasp_rectangle = [str2num(point0),
                               str2num(point1),
                               str2num(point2),
                               str2num(point3)]
            grasp_rectangles.append(grasp_rectangle)

    return grasp_rectangles
i=90
grs = get_rectangle(label[i])
print(grs)
import cv2
import random
import cv2
img = cv2.imread(rgb[i])
for gr in grs:
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for i in range(3):
        img = cv2.line(img, gr[i], gr[i + 1], color, 3)
    img = cv2.line(img, gr[3], gr[0], color, 2)

plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.show()
plt.imshow(img)
plt.show()
