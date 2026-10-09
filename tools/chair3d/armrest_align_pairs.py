import cv2,numpy as np,sys
from PIL import Image,ImageOps
S=sys.argv[1]
L=lambda f:np.ascontiguousarray(np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB'))[...,::-1])
def align(ref,mov,excl):
    gr=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY);gm=cv2.cvtColor(mov,cv2.COLOR_BGR2GRAY)
    sc=0.35; r=cv2.resize(gr,None,fx=sc,fy=sc);m=cv2.resize(gm,None,fx=sc,fy=sc)
    mask=np.full(r.shape,255,np.uint8)
    for (x0,y0,x1,y1) in excl: mask[int(y0*sc):int(y1*sc),int(x0*sc):int(x1*sc)]=0
    orb=cv2.ORB_create(8000);k1,d1=orb.detectAndCompute(r,mask);k2,d2=orb.detectAndCompute(m,mask)
    mt=cv2.BFMatcher(cv2.NORM_HAMMING,True).match(d1,d2)
    p1=np.float32([k1[x.queryIdx].pt for x in mt])/sc;p2=np.float32([k2[x.trainIdx].pt for x in mt])/sc
    H,inl=cv2.findHomography(p2,p1,cv2.RANSAC,4.0);print('inliers',inl.sum(),'/',len(mt))
    return cv2.warpPerspective(mov,H,(ref.shape[1],ref.shape[0]),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
def crop(im,y0):  # 16:9 landscape crop, full width
    return cv2.resize(im[y0:y0+1899],(1920,1080),interpolation=cv2.INTER_AREA)
C='11_usp_armrest_closeups/';D='12_usp_armrest_4D_positions/'
arm_excl=[(1050,1500,2700,4800)]   # armrest zone in closeups (full res)
pairs={'vertical':(C+'02.jpg',C+'14.jpg',arm_excl,1650),
       'diagonal':(C+'26.jpg',C+'32.jpg',arm_excl,1650),
       'horizontal':(D+'34.jpg',D+'42.jpg',[(150,1700,2300,3800)],1750)}
rows=[]
for k,(a,b,ex,y0) in pairs.items():
    A=L(a);B=align(A,L(b),ex)
    ca,cb=crop(A,y0),crop(B,y0)
    cv2.imwrite(f'{S}/{k}_start.png',ca);cv2.imwrite(f'{S}/{k}_end.png',cb)
    t=[cv2.resize(x,(640,360)) for x in (ca,cb)]
    cv2.putText(t[0],k+' start',(10,30),0,1,(0,0,255),2);cv2.putText(t[1],k+' end',(10,30),0,1,(0,0,255),2)
    rows.append(cv2.hconcat(t))
cv2.imwrite(S+'/pairs.jpg',cv2.vconcat(rows))
