import cv2,numpy as np,sys,subprocess
S=sys.argv[1]
st=np.load(S+'/stab.npy')
xs=[]
for o in st:
    gg=cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)[700:900,900:1800]
    xs.append(np.where((gg<90).sum(0)>40)[0].max())
xs=np.array(xs,float)
# unique frames 17..60, drop pulldown duplicates
idx=[17]
for i in range(18,61):
    if np.abs(st[i].astype(np.int16)-st[idx[-1]].astype(np.int16)).mean()>0.6: idx.append(i)
d=np.maximum.accumulate(xs[idx]-xs[idx[0]]); D=d[-1]
print(len(idx),'unique, travel',D)
ss=lambda u:u*u*u*(u*(6*u-15)+10)  # smootherstep
seq=[0]*8+[ss(k/35) for k in range(36)]+[1]*14+[ss(1-k/29) for k in range(30)]+[0]*2
pick=[idx[int(np.argmin(np.abs(d-s*D)))] for s in seq]
print(len(seq),pick)
sc=1.035
M=cv2.getRotationMatrix2D((960,540),0,sc)
p=subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1080','-r','30','-i','-','-c:v','libx264','-crf','14','-preset','slow','-pix_fmt','yuv420p','-movflags','+faststart',S+'/usp4_from_clip.mp4'],stdin=subprocess.PIPE)
for i in pick:
    p.stdin.write(cv2.warpAffine(st[i],M,(1920,1080),flags=cv2.INTER_LANCZOS4).tobytes())
p.stdin.close();p.wait()
