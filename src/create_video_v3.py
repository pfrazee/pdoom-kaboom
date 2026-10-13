from PIL import Image,ImageDraw,ImageFont
import numpy as np
import math,json,bisect,subprocess,time,sys
from functools import lru_cache
import create_video_v2 as v
base=v.base;ROOT=v.ROOT;WORK=v.WORK;OUT=v.OUT
W,H,S,FPS=1920,1080,1.5,30
DURATION=151.6
TRACK=np.load(ROOT/'data/v3-mouth-track.npz');MOUTH=TRACK['open'];ENERGY=TRACK['energy']
CUES=json.loads((ROOT/'data/v3-cues.json').read_text());STARTS=[c['start'] for c in CUES]
# Timing is audio-derived at 10 ms resolution. No sine wave or per-character lip offset.
def mouth_at(t):return bool(MOUTH[min(len(MOUTH)-1,max(0,round(t*100)))])
def energy_at(t):return float(ENERGY[min(len(ENERGY)-1,max(0,round(t*100)))])

def ease(t):return base.ease(t)
def rot(rx=0,ry=0,rz=0):
 cx,sx=math.cos(rx),math.sin(rx);cy,sy=math.cos(ry),math.sin(ry);cz,sz=math.cos(rz),math.sin(rz)
 return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

v.CAMERAS.update({'portrait':((0,273,800),(0,210,70),1120),'direct':((0,230,665),(0,211,70),1190),'studio':((0,315,1130),(0,208,30),1310),'duet':((0,260,1000),(0,205,30),1280)})
v.CAMS.update({name:v.Camera(name) for name in ['portrait','direct','studio','duet']})
# start, set, camera, performance. Camera cuts, entrances, and gestures are choreographed independently.
SHOTS=[
 (0,'stage','wide','welcome'),(1.35,'slide','portrait','skid'),(5.18,'stage','left','present'),
 (8.80,'solo','direct','lean'),(11.86,'spot','portrait','shrug'),
 (15.05,'pop','portrait','question'),(19.02,'duet','duet','debate'),(22.53,'stage','right','point'),
 (26.20,'spot','direct','lean'),(29.50,'trio','studio','hat_tip'),
 (31.85,'stage','wide','kick'),(36.09,'trio','studio','kick'),(39.44,'solo','portrait','point'),
 (42.87,'stage','low','punch'),(46.63,'parade','studio','walk'),
 (49.65,'slide_right','portrait','skid'),(52.92,'solo','portrait','hat_tip'),
 (56.67,'stage','left','present'),(60.10,'spot','direct','lean'),
 (63.45,'duet','duet','debate'),(66.65,'pop','portrait','question'),
 (70.71,'stage','right','shrug'),(74.06,'spot','direct','lean'),
 (78.93,'stage','wide','kick'),(83.08,'trio','studio','kick'),(86.51,'solo','direct','point'),
 (89.94,'stage','low','punch'),(94.10,'parade','studio','walk'),
 (96.40,'slide','portrait','skid'),(100.15,'solo','direct','lean'),
 (103.82,'duet','duet','debate'),(107.57,'spot','portrait','hat_tip'),
 (111.6,'stage','low','present'),(114.00,'trio','studio','hat_tip'),
 (116.98,'stage','wide','kick'),(121.21,'trio','studio','kick'),
 (124.72,'solo','direct','punch'),(128.15,'stage','low','punch'),
 (131.43,'spot','direct','lean'),(135.08,'duet','duet','punch'),
 (139.70,'solo','portrait','present'),(144.00,'spot','direct','lean'),
 (147.45,'stage','wide','bow')]
SHOT_STARTS=[s[0] for s in SHOTS]
(WORK/'v3-shot-list.json').write_text(json.dumps([dict(zip(['start','set','camera','pose'],s)) for s in SHOTS],indent=2))

class Performer(v.Actor):
 def __init__(self,sc,x,z,scale,t,ident,mode,u,cam,move_y=0):
  super().__init__(sc,x,z,scale,t,ident,mode in ['kick','punch','walk'])
  self.mode=mode;self.u=u;self.pitch=0;self.stretch=1;self.move_y=move_y
  phase=t*2.27*math.tau;alt=math.sin(phase*.5+ident*.5)
  self.roll=.025*alt;self.yaw=math.atan2(cam.pos[0]-x,cam.pos[2]-z)*.4
  self.bob=max(0,math.sin(phase))*(6 if ident==0 else 4)
  self.armL=-.15;self.armR=.15;self.hat_lift=0;self.hat_tip=0
  if mode in ['present','welcome']:
   self.armL=-.6-.25*math.sin(u*2);self.armR=.8+.25*math.sin(u*2+1)
   self.roll=.05*math.sin(u*1.2);self.yaw+=.12*math.sin(u)
  elif mode in ['shrug','question','debate']:
   pulse=.5+.5*math.sin(u*2.4+ident*.6)
   self.armL=-.55-.6*pulse;self.armR=.55+.6*pulse
   self.roll=.13*math.sin(u*2+ident);self.bob+=6*pulse
  elif mode=='point':
   self.armL=-.2;self.armR=1.25+.18*math.sin(u*2)
   self.roll=-.09;self.yaw+=.17*math.sin(u)
  elif mode=='lean':
   self.roll=.13*math.sin(u*1.6);self.pitch=.10+.06*math.sin(u*1.4)
   self.armL=-.35-.25*math.sin(u*1.4);self.armR=.35+.25*math.sin(u*1.4+1)
   self.bob+=4*energy_at(t)
  elif mode=='hat_tip':
   tip=max(0,math.sin(min(u,2.1)/2.1*math.pi))
   self.armL=-1.25*tip;self.armR=.45;self.hat_lift=16*tip;self.hat_tip=.28*tip
   self.roll=-.06*tip;self.pitch=.16*tip
  elif mode in ['kick','punch']:
   self.bob=max(0,math.sin(phase))*25
   self.armL=-.65-.45*math.sin(phase*.5);self.armR=.65+.45*math.sin(phase*.5)
   if mode=='punch':self.armL=-1.35;self.armR=1.35;self.bob+=6
   self.roll=.09*math.sin(phase*.5);self.yaw+=.15*math.sin(phase*.5)
   self.stretch=1+.025*math.sin(phase)
  elif mode=='walk':
   self.armL=-.45-.5*math.sin(phase);self.armR=.45-.5*math.sin(phase)
   self.roll=.075*math.sin(phase*.5);self.yaw=.25;self.bob=max(0,math.sin(phase))*11
  elif mode=='skid':
   p=max(0,1-u/.9);self.roll=.24*p;self.armL=-.7*p;self.armR=1.05*p+.25
   self.yaw+=-.25*p;self.bob=6*math.sin(u*9)*math.exp(-u*3)
  elif mode=='bow':
   p=ease((u-.10)/1.15);self.pitch=.50*p;self.armL=-.15;self.armR=.2
   self.hat_lift=12*p;self.hat_tip=.16*p;self.bob=0
  self.bodyR=rot(self.pitch,self.yaw,self.roll)
  self.partR=np.eye(3);self.pivot=np.zeros(3);self.offset=np.zeros(3)
 def part(self,pivot=(0,0,0),rx=0,ry=0,rz=0,offset=(0,0,0)):
  self.partR=rot(rx,ry,rz);self.pivot=np.array(pivot);self.offset=np.array(offset)
 def tr(self,pts):
  p=(np.asarray(pts,dtype=float)-self.pivot)@self.partR.T+self.pivot+self.offset
  p[:,1]=(p[:,1]-105)*self.stretch+105
  p=(p-np.array([0,105,0]))@self.bodyR.T+np.array([0,105,0])
  p*=self.scale;p+=np.array([self.x,self.bob+self.move_y,self.z]);return p


def character(sc,t,x,z,scale,ident,mode,u,cam,sings=True,move_y=0):
 a=Performer(sc,x,z,scale,t,ident,mode,u,cam,move_y)
 phase=t*2.27*math.tau
 for i,x0 in enumerate([-108,-59,33,82]):
  beat=math.sin(phase+(i%2)*math.pi+ident*.35)
  kick=max(0,beat)*(.70 if mode in ['kick','punch'] else (.30 if mode=='walk' else .07))
  rz=(1 if i<2 else -1)*(.11 if mode in ['kick','skid'] else 0)
  a.part(pivot=(x0+14,66,0),rx=-kick,rz=rz)
  a.box(x0,x0+28,0,66,-23,28,top=False)
 for side,angle in [(-1,a.armL),(1,a.armR)]:
  a.part(pivot=(side*110,165,0),rz=angle)
  lo,hi=(-156,-106) if side<0 else (106,156)
  a.box(lo,hi,137,192,-24,25,'#ed8647')
 a.part();a.box(-110,110,66,257,-26,30,None)
 squint=(mode in ['kick','punch'] and math.sin(phase*.25+ident)>.9) or math.sin(t*1.1+ident*2.1)>.996
 sc.texture(a.tr(v.plane(-110,110,66,257,30.1)),v.face_texture(squint,bool(sings and mouth_at(t))))
 a.part(pivot=(0,268,0),rz=a.hat_tip,offset=(0,a.hat_lift,0))
 a.cylinder(137,70,265,272,'#dab56d','#aa7e40');a.ellipse(137,70,272.4,'#e8c780')
 for inset in [3,8,14,20]:a.ring(137-inset,70-inset*.5,272.7,'#b18c4c',.65)
 a.cylinder(84,49,274,277,'#e7c681','#b78b4c');a.cylinder(84,49,297,327,'#e7c681','#b78b4c')
 for yy in range(299,325,5):a.ring(84.2,49.2,yy,'#bb9456',.6)
 for ang in np.linspace(0,math.tau,49):a.line([(84.3*math.cos(ang),299,49.3*math.sin(ang)),(84.3*math.cos(ang),325,49.3*math.sin(ang))],'#f7df9e',.65)
 a.cylinder(85,50,277,297,'#862d41','#522333');a.ring(85.2,50.2,295,'#bd4e58',1.5)
 a.ellipse(84,49,327.5,'#f1d696')
 for inset in range(5,70,9):a.ring(84-inset,49-inset*.54,327.8,'#c5a163',.7)

@lru_cache(maxsize=24)
def studio_background(kind,theme):
 yy,xx=np.mgrid[0:H,0:W];xx=xx/S;yy=yy/S
 palettes={'plum':('#22182d','#674050'),'teal':('#112c38','#216478'),'cream':('#b89165','#ecce94'),'spot':('#090c15','#272235')}
 dark,light=palettes[theme]
 glow=np.clip(1-np.sqrt(((xx-640)/750)**2+((yy-300)/570)**2),0,1)[...,None]
 arr=np.array(v.hexrgb(dark))+glow*(np.array(v.hexrgb(light))-np.array(v.hexrgb(dark)))
 im=Image.fromarray(np.uint8(arr));d=ImageDraw.Draw(im)
 if kind=='duet':
  base.rect(d,(0,0,638,635),'#173d4c');base.rect(d,(642,0,1280,635),'#533145');base.rect(d,(638,0,642,635),'#deb67d')
 elif kind=='trio':
  base.rect(d,(0,0,424,635),'#173d4c');base.rect(d,(428,0,852,635),'#583245');base.rect(d,(856,0,1280,635),'#254643')
  for x in [426,854]:base.line(d,[(x,0),(x,635)],'#d5ad74',2)
 else:
  ov=Image.new('RGBA',im.size);od=ImageDraw.Draw(ov)
  if kind=='spot':
   base.poly(od,[(594,0),(686,0),(1030,616),(250,616)],(248,220,162,12))
   base.ellipse(od,(335,528,945,618),(254,224,161,21))
  else:
   base.ellipse(od,(330,20,950,616),(249,225,176,17))
   base.ellipse(od,(339,29,941,607),None,(240,206,156,44),1)
  im=Image.alpha_composite(im.convert('RGBA'),ov).convert('RGB')
 return im

@lru_cache(maxsize=100)
def caption(idx):
 tx=CUES[idx]['text'];sz=29
 while base.F(sz).getlength(tx)>1140*S and sz>26:sz-=1
 lines=[tx]
 if base.F(sz).getlength(tx)>1140*S:
  words=tx.split();k=min(range(1,len(words)),key=lambda i:abs(base.F(sz).getlength(' '.join(words[:i]))-base.F(sz).getlength(' '.join(words[i:]))));lines=[' '.join(words[:k]),' '.join(words[k:])]
 cap=Image.new('RGBA',(W,126));d=ImageDraw.Draw(cap)
 for i,tx in enumerate(lines):base.text(d,(640,(36 if len(lines)==1 else 23)+33*i),tx,sz,'#fff5dc',stroke=1,stroke_fill='#16141e')
 return cap


def render(t,override=None):
 shotidx=max(0,bisect.bisect_right(SHOT_STARTS,t)-1)
 st,kind,camname,mode=SHOTS[shotidx] if override is None else override
 end=SHOTS[shotidx+1][0] if shotidx+1<len(SHOTS) else DURATION;u=t-st
 idx=bisect.bisect_right(STARTS,t)-1;c=CUES[idx] if idx>=0 and t<CUES[idx]['end'] else None
 sec=CUES[idx]['section'] if idx>=0 else 'Intro';chorus=('Chorus' in sec and 'Pre' not in sec) or sec=='Outro'
 theme='gold' if chorus else ('purple' if 'Pre' in sec else ('teal' if sec=='Bridge' else 'amber'))
 cam=v.CAMS[camname]
 im=(v.background(camname,theme) if kind=='stage' else studio_background(kind,'spot' if kind=='spot' else ('teal' if sec=='Bridge' else ('plum' if chorus or 'Pre' in sec else 'cream')))).copy()
 # x,z,size,id,gesture,local_time,sings,vertical_offset
 cast=[]
 if kind=='stage':
  for x,ident in [(-470,1),(-230,2),(230,3),(470,4)]:
   cast.append((x+14*math.sin(t*2.27*math.pi+ident*.2),-180,.80,ident,mode if chorus or mode=='bow' else 'present',u+ident*.12,chorus,0))
  cast.append((0,75,1.,0,mode,u,True,0))
 elif kind in ['slide','slide_right']:
  direction=1 if kind=='slide_right' else -1
  p=ease(u/.62);xx=direction*820*(1-p)+18*math.sin(max(0,u-.62)*10)*math.exp(-max(0,u-.62)*5)
  cast=[(xx,75,1.06,0,'skid' if u<1.1 else 'present',u,True,0)]
 elif kind=='pop':
  rise=-520*(1-ease(u/.44))+16*math.sin(max(0,u-.44)*9)*math.exp(-max(0,u-.44)*4)
  cast=[(0,75,1.06,0,mode,u,True,rise)]
 elif kind=='duet':
  active=idx%2
  for i,x in enumerate([-258,258]):cast.append((x,75,1.0,i,mode,u+i*.27,chorus or i==active,0))
 elif kind=='trio':
  for i,x in enumerate([-358,0,358]):cast.append((x,75,.88,i,mode,u+i*.17,(chorus or i==1),0))
 elif kind=='parade':
  travel=-580+1160*ease(min(1,u/max(3.3,end-st)))
  for i in range(3):cast.append((travel-i*260,75-i*40,.88,i,'walk',u,(i==0 or chorus),0))
 else:
  z=75+(32*ease(u/max(1,end-st)) if camname=='direct' else 0)
  cast=[(0,z,1.02,0,mode,u,True,0)]
 im=v.shadows(im,cam,t,[(x,z,s,i) for x,z,s,i,_,_,_,_ in cast]);sc=v.Scene(cam)
 for x,z,s,ident,gesture,local,sings,move_y in cast:character(sc,t,x,z,s,ident,gesture,local,cam,sings,move_y)
 if kind=='stage':v.microphone(sc)
 sc.paint(im)
 if (42.926<t<46.676) or (90<t<94) or (128.218<t<135.08) or t>147.4:
  d=ImageDraw.Draw(im)
  for x,y,vel,p,col in base.CONFETTI:
   yy=(y+t*vel)%870-145;xx=x+32*math.sin(t*1.8+p)
   if yy>630:continue
   sz=2.5;aa=t*2.1+p
   base.poly(d,[(xx+math.cos(aa)*sz,yy+math.sin(aa)*sz),(xx+math.cos(aa+1.57)*sz*.5,yy+math.sin(aa+1.57)*sz*.5),(xx-math.cos(aa)*sz,yy-math.sin(aa)*sz),(xx-math.cos(aa+1.57)*sz*.5,yy-math.sin(aa+1.57)*sz*.5)],col)
 if t>149:
  d=ImageDraw.Draw(im);edge=640*ease((t-149)/2.45)
  for side in [0,1]:
   for x in range(round(edge)):
    val=.5+.38*math.cos(x*.12);xx=x if not side else 1280-x;base.rect(d,(xx,0,xx+1,635),(int(80+59*val),int(21+15*val),int(35+20*val)))
   xx=edge if not side else 1280-edge;base.line(d,[(xx,0),(xx,635)],'#d4ad70',2)
 d=ImageDraw.Draw(im);base.rect(d,(0,635,1280,720),'#15151f');base.line(d,[(44,635),(1236,635)],'#6e5946',.8)
 if c:
  cap=caption(idx);im.paste(cap,(0,954),cap);d=ImageDraw.Draw(im)
  p=max(0,min(1,(t-c['start'])/(c['end']-c['start'])));base.line(d,[(570,710),(710,710)],'#46404a',2);base.line(d,[(570,710),(570+140*p,710)],'#e7bd76',2)
 for x in [35,1245]:base.poly(d,[(x,674),(x+4,679),(x,684),(x-4,679)],'#bf945c')
 fade=min(1,t/.35,(DURATION-t)/.7)
 if fade<1:im=Image.blend(Image.new('RGB',im.size,'#090b13'),im,max(0,fade))
 return im


def render_segment(start,end,dest):
 cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','3','-movflags','+faststart',str(dest)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE);first=round(start*FPS);last=round(end*FPS);st=time.time()
 try:
  for i in range(first,last):
   p.stdin.write(render(i/FPS).tobytes())
   if (i-first)%300==0:print(f'{dest.name}: {i-first}/{last-first} frames ({time.time()-st:.1f}s)',flush=True)
 finally:p.stdin.close()
 if p.wait()!=0:raise RuntimeError('Encoder failed')
 print(f'COMPLETE {dest.name} {time.time()-st:.1f}s',flush=True)

if __name__=='__main__':
 if sys.argv[1]=='stills':
  times=[1.67,2.6,9.5,12.6,15.7,20.3,33.0,37.2,48.4,54,64.5,108.7]
  sheet=Image.new('RGB',(1440,1080),'#171420')
  for i,t in enumerate(times):
   im=render(t);im.save(WORK/f'v3-frame-{t}.jpg',quality=95);sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),(i%3*480,i//3*270))
  sheet.save(WORK/'v3-contact.jpg',quality=95)
 elif sys.argv[1]=='segment':render_segment(float(sys.argv[2]),float(sys.argv[3]),WORK/sys.argv[4])
