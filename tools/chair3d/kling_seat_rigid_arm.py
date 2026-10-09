import cv2,numpy as np,sys,subprocess
S=sys.argv[1]; OUT=sys.argv[2]; PREVIEW=len(sys.argv)>3
fr=np.load(S+'/k6.npy'); A,B=46,72
g=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in fr]
xs=np.array([np.where((x[700:900,900:1850]<90).sum(0)>40)[0].max() for x in g],float)
idx=list(range(A,B+1)); d=np.maximum.accumulate(xs[idx]-xs[A]); D=float(d[-1])
# moving unit (seat + shell + both armrests) in frame 46, excludes backrest/lumbar ring
P=np.array([(775,180),(1275,180),(1300,240),(1300,300),(1075,305),(1010,375),(1345,375),(1365,440),(1360,505),(1100,510),(1040,575),(1330,590),(1530,650),(1585,800),(1555,920),(1470,975),(1300,1012),(860,1035),(840,1052),(640,1052),(622,985),(442,955),(434,880),(434,800),(452,786),(535,725),(610,672),(668,604),(722,592),(740,420),(748,300)],np.int32)
M=np.zeros((1080,1920),np.float32);cv2.fillPoly(M,[P],1.0);M=cv2.GaussianBlur(M,(0,0),1.5)[...,None]
src=fr[A].astype(np.float32)
ss=lambda u:u*u*u*(u*(6*u-15)+10)
seq=[0]*8+[ss(k/35) for k in range(36)]+[1]*14+[ss(1-k/29) for k in range(30)]+[0]*2
if PREVIEW: seq=[0,0.5,1.0]
outs=[]
for s in seq:
    dx=s*D
    j=idx[int(np.argmin(np.abs(d-dx)))]           # Kling frame with same travel: supplies revealed background
    T=np.float32([[1,0,dx],[0,1,0]])
    mv=cv2.warpAffine(src,T,(1920,1080),flags=cv2.INTER_CUBIC)
    mm=cv2.warpAffine(M[...,0],T,(1920,1080))[...,None]
    base=fr[j].astype(np.float32)
    o=base*(1-mm)+mv*mm
    outs.append(np.clip(o,0,255).astype(np.uint8))
if PREVIEW:
    cv2.imwrite(OUT,cv2.vconcat([cv2.hconcat([cv2.resize(outs[i],(960,540)) for i in (0,1)]),cv2.hconcat([cv2.resize(outs[2],(960,540)),cv2.resize(fr[B],(960,540))])]))
    cv2.imwrite(OUT.replace('.jpg','_end.png'),outs[2]); sys.exit()
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r','30','-i','-','-an','-c:v','libx264','-crf','14','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',OUT],stdin=subprocess.PIPE)
for o in outs:
    bl=cv2.GaussianBlur(o,(0,0),1.2);p.stdin.write(cv2.addWeighted(o,1.5,bl,-0.5,0).tobytes())
p.stdin.close();p.wait()
