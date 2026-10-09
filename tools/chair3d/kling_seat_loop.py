import cv2,numpy as np,sys,subprocess
F,OUT=sys.argv[1],sys.argv[2]
A,B=int(sys.argv[3]),int(sys.argv[4])
c=cv2.VideoCapture(F);fr=[]
while True:
    ok,f=c.read()
    if not ok:break
    fr.append(f)
g=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in fr]
xs=np.array([np.where((x[700:900,900:1850]<90).sum(0)>40)[0].max() for x in g],float)
idx=list(range(A,B+1))
d=np.maximum.accumulate(xs[idx]-xs[A]);D=d[-1]
ss=lambda u:u*u*u*(u*(6*u-15)+10)
seq=[0]*8+[ss(k/35) for k in range(36)]+[1]*14+[ss(1-k/29) for k in range(30)]+[0]*2
pick=[idx[int(np.argmin(np.abs(d-s*D)))] for s in seq]
print('travel',D,pick)
k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-crf','14','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
for i in pick:
    f=fr[i];bl=cv2.GaussianBlur(f,(0,0),1.2);f=cv2.addWeighted(f,1.5,bl,-0.5,0)  # mild unsharp
    p.stdin.write(f.tobytes())
p.stdin.close();p.wait()
