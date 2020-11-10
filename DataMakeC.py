#说明：先修正大小，后根据规则产生数据
#
# #产生数据##########################################################################################
# from PIL import Image
# import random
# #from pylab import *
#
# for index in range(1,2001):############################372 or 2001
#     img = Image.open("/Users/HZK/YBJ/image_recovery/labelC/dog_" + str(index) + ".jpg")###############
#     #img = I.convert('L')  # rgb->gray
#     #img = Image.open("/Users/HZK/YBJ/image_recovery/trainC/dog_"+str(index)+".jpg")##############
#     list_x = []
#     for k in range(0,img.size[0]):
#         list_x.append(k)
#     for j in range(0,img.size[1]):
#         for jk in range(0, 3):
#             slice = random.sample(list_x, int(0.6 * img.size[0]))#################0.8?
#             for i in range(0,img.size[0]):
#                 data = (img.getpixel((i, j)))
#                 #print (type(data))
#                 data = list(data)
#                 if i in slice:
#                     data[jk] = 0
#                     img.putpixel((i,j),(data[0],data[1],data[2],255))
#     img = img.convert("RGB")
#     img.save("/Users/HZK/YBJ/image_recovery/trainC/dog_" + str(index) + ".jpg")########################
###########################################################################################################

# #处理图片大小
from PIL import Image
import os.path
import glob

def convertjpg(jpgfile,outdir,width=264,height=400):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

for jpgfile in glob.glob("/Users/HZK/YBJ/image_recovery/labelB/*.jpg"):
    convertjpg(jpgfile,"/Users/HZK/YBJ/image_recovery/labelC1/")
    ####################################################################