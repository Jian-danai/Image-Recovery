from PIL import Image
import os.path
import glob

def convertpng(pngfile,outdir,width=306,height=408):
    img=Image.open(pngfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
    except Exception as e:
        print(e)

#需要改路径！！
for pngfile in glob.glob("C:\\Users\\Administrator\\Desktop\\homework\\resultA\\*.png"):
    convertpng(pngfile,"C:\\Users\\Administrator\\Desktop\\homework\\result_tran\\")