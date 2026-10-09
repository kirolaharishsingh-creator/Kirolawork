import cv2,numpy as np,sys
F,S=sys.argv[1],sys.argv[2]
c=cv2.VideoCapture(F);fr=[]
while True:
    ok,f=c.read()
    if not ok:break
    fr.append(f)
g=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32) for f in fr]
# static patches: backrest frame/headrest/lumbar, left of armrest & seat
pts=[]
for y in range(40,1000,60):
    for x in range(40,560,60):
        if y>560 and x>400: continue
        p=g[0][y-30:y+30,x-30:x+30]
        if p.std()>18: pts.append((x,y))
pts=np.float32(pts);print(len(pts),'patches')
A=[]
prev=np.zeros_like(pts)
for i in range(len(g)):
    q=[]
    for (x,y),pv in zip(pts.astype(int),prev):
        t=g[0][y-30:y+30,x-30:x+30];cx,cy=int(x+pv[0]),int(y+pv[1]);R=24
        win=g[i][max(cy-30-R,0):cy+30+R,max(cx-30-R,0):cx+30+R]
        r=cv2.matchTemplate(win,t,cv2.TM_SQDIFF_NORMED);_,_,mn,_=cv2.minMaxLoc(r)
        q.append((max(cx-30-R,0)+mn[0]+30,max(cy-30-R,0)+mn[1]+30))
    q=np.float32(q);prev=q-pts
    M,inl=cv2.estimateAffine2D(pts,q,ransacReprojThreshold=2)
    A.append(M.ravel())
A=np.array(A)
As=np.array([np.convolve(np.pad(A[:,j],3,mode='edge'),np.ones(7)/7,'valid') for j in range(6)]).T
print(np.round(A[[0,30,60]],3))
out=[cv2.warpAffine(f,cv2.invertAffineTransform(As[i].reshape(2,3)),(1920,1080),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE) for i,f in enumerate(fr)]
np.save(S+'/stab.npy',np.array(out))
