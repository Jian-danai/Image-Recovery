from PIL import Image
import os.path
import glob

def convertpng(pngfile,outdir,width=266,height=402):
    img=Image.open(pngfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
    except Exception as e:
        print(e)

for pngfile in glob.glob("C:\\Users\\Administrator\\Desktop\\homework\\resultC\\*.png"):
    convertpng(pngfile,"C:\\Users\\Administrator\\Desktop\\homework\\result_tran\\")