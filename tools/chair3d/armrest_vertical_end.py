import cv2,numpy as np,sys
S=sys.argv[1]; R=int(sys.argv[2]) if len(sys.argv)>2 else 350
a=np.load(S+'/a02.npy').astype(np.float32)
g=cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_BGR2GRAY)
X0,X1,Y0,Y1=1215,2370,2210,3300
# pad + tube-top mask (moving part)
roi=g[Y0:Y1,X0:X1]
wall=np.median(g[2000:2200,1300:2300]);print('wall',wall)
m=np.zeros(g.shape,np.uint8);m[Y0:Y1,X0:X1]=(roi<wall-25).astype(np.uint8)
m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
n,lab,st,_=cv2.connectedComponentsWithStats(m);k=1+np.argmax(st[1:,4]);m=(lab==k).astype(np.uint8)
m=cv2.dilate(m,np.ones((7,7),np.uint8))  # include anti-aliased rim
mf=cv2.GaussianBlur(m.astype(np.float32),(0,0),2)[...,None]
# wall plate: row-wise linear between left/right wall samples + matched grain
plate=a.copy()
yy=np.arange(Y0-20,Y1+20)
L=a[yy,1180:1210].mean(1);Rt=a[yy,2380:2410].mean(1)
t=np.linspace(0,1,X1+20-(X0-20))[None,:,None]
fill=L[:,None,:]*(1-t)+Rt[:,None,:]*t
grain=a[2000:2200,1300:2300]-cv2.GaussianBlur(a[2000:2200,1300:2300],(0,0),8)
gs=np.tile(grain,(int(np.ceil(fill.shape[0]/200))+1,2,1))[:fill.shape[0],:fill.shape[1]]
fill=fill+gs
reg=(slice(Y0-20,Y1+20),slice(X0-20,X1+20))
mm=cv2.dilate(m,np.ones((25,25),np.uint8)).astype(np.float32)
mm=cv2.GaussianBlur(mm,(0,0),6)[...,None][reg]
plate[reg]=plate[reg]*(1-mm)+fill*mm
# tube: stretch the real post section [3240,3560] up to start at 3240-R (keeps taper, no seam)
tube_cols=np.where(g[3350,1300:2100]<wall-25)[0]+1300;tx0,tx1=tube_cols.min()-30,tube_cols.max()+30;print('tube',tx0,tx1)
TA,TB=3240,3560
ys=np.arange(TA-R,TB)
srcy=(TA+(ys-(TA-R))*(TB-TA)/(TB-TA+R)).astype(np.float32)
band=cv2.remap(a[:,tx0:tx1],np.tile(np.arange(tx1-tx0,dtype=np.float32),(len(ys),1)),np.tile(srcy[:,None],(1,tx1-tx0)),cv2.INTER_CUBIC)
w=np.ones(tx1-tx0,np.float32);w[:12]=np.linspace(0,1,12);w[-12:]=np.linspace(1,0,12)
plate[TA-R:TB,tx0:tx1]=plate[TA-R:TB,tx0:tx1]*(1-w[None,:,None])+band*w[None,:,None]
# paste moved pad (shifted up by R)
M=np.float32([[1,0,0],[0,1,-R]])
mv=cv2.warpAffine(a,M,(a.shape[1],a.shape[0]));mvm=cv2.warpAffine(mf[...,0],M,(a.shape[1],a.shape[0]))[...,None]
out=plate*(1-mvm)+mv*mvm
out=np.clip(out,0,255).astype(np.uint8)
cv2.imwrite(S+'/v_end_full.png',out)
cv2.imwrite(S+'/v_cmp.jpg',cv2.hconcat([cv2.resize(a[1500:4300,700:2700].astype(np.uint8),(714,1000)),cv2.resize(out[1500:4300,700:2700],(714,1000))]))
