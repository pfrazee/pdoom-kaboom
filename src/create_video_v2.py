from PIL import Image, ImageDraw
import numpy as np
import math, subprocess, sys, time, bisect, json
from pathlib import Path
from functools import lru_cache
import create_video as base

ROOT=base.ROOT; WORK=base.WORK; OUT=base.OUT
W,H,S,FPS=1920,1080,1.5,24
DURATION=151.6
CUES=base.CUES; STARTS=base.STARTS; env=base.env

# Camera positions are in the same three-dimensional space as the stage and cast.
CAMERAS={
 'wide':((0,330,1080),(0,210,-20),1250),
 'left':((-490,315,940),(-40,210,20),1340),
 'right':((490,300,920),(40,214,20),1330),
 'close':((38,267,690),(0,210,65),1190),
 'medium':((-80,285,860),(0,213,45),1230),
 'low':((125,112,790),(0,193,65),1100),
 'high':((-125,730,1120),(0,163,-15),1350),
 'chorus_left':((-445,242,645),(-430,158,-175),1440),
 'chorus_right':((445,242,645),(430,158,-175),1440),
}
SHOTS=[
 (0,'wide'),(5.186,'left'),(8.856,'close'),(11.888,'right'),
 (15.479,'low'),(19.069,'medium'),(22.580,'left'),(26.250,'close'),(29.700,'high'),
 (31.915,'wide'),(36.144,'chorus_left'),(39.495,'right'),(42.926,'low'),(46.676,'high'),
 (49.900,'medium'),(52.979,'left'),(56.729,'close'),(60.160,'right'),
 (63.511,'medium'),(67.021,'close'),(70.771,'left'),(74.122,'low'),
 (78.989,'wide'),(83.138,'chorus_right'),(86.569,'left'),(90.000,'close'),(94.000,'high'),
 (96.463,'medium'),(100.213,'right'),(103.883,'close'),(107.633,'left'),(112.000,'low'),
 (117.048,'wide'),(121.277,'chorus_left'),(124.787,'chorus_right'),(128.218,'low'),
 (131.500,'close'),(135.160,'right'),(139.787,'close'),(144.000,'high'),(147.400,'wide')
]
SHOT_STARTS=[t for t,_ in SHOTS]
(WORK/'v2-shot-list.json').write_text(json.dumps([{'start':t,'camera':c} for t,c in SHOTS],indent=2))

class Camera:
 def __init__(self,name):
  position,target,focal=CAMERAS[name]
  self.pos=np.array(position,dtype=float); fw=np.array(target,dtype=float)-self.pos;fw/=np.linalg.norm(fw)
  right=np.cross(fw,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,fw)
  self.mat=np.stack([right,up,fw],axis=1);self.f=focal
 def project(self,pts):
  xyz=(np.asarray(pts)-self.pos)@self.mat
  return np.stack([640+xyz[:,0]/xyz[:,2]*self.f,317-xyz[:,1]/xyz[:,2]*self.f],axis=1),xyz[:,2]

CAMS={k:Camera(k) for k in CAMERAS}

class Scene:
 def __init__(self,camera):self.cam=camera;self.items=[]
 def poly(self,pts,col):
  points,z=self.cam.project(pts)
  if min(z)<1:return
  self.items.append((float(np.mean(z)),0,points,col,0))
 def line(self,pts,col,width=1):
  points,z=self.cam.project(pts)
  if min(z)<1:return
  self.items.append((float(np.mean(z))-.04,1,points,col,width))
 def texture(self,pts,texture):
  points,z=self.cam.project(pts)
  self.items.append((float(np.mean(z)),2,points,texture,0))
 def paint(self,im):
  d=ImageDraw.Draw(im)
  for z,kind,points,col,width in sorted(self.items,key=lambda i:i[0],reverse=True):
   p=[(round(x*S),round(y*S)) for x,y in points]
   if kind==2:
    xmin=max(0,min(x for x,y in p));ymin=max(0,min(y for x,y in p))
    xmax=min(W,max(x for x,y in p)+1);ymax=min(H,max(y for x,y in p)+1)
    if xmax<=xmin or ymax<=ymin:continue
    src=[(0,col.height),(col.width,col.height),(col.width,0),(0,0)]
    mat=[];vals=[]
    for (xx,yy),(u,v) in zip(p,src):
     x=xx-xmin;y=yy-ymin
     mat += [[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]];vals += [u,v]
    co=np.linalg.solve(np.array(mat),np.array(vals))
    patch=col.transform((xmax-xmin,ymax-ymin),Image.Transform.PERSPECTIVE,co,Image.Resampling.BICUBIC)
    im.paste(patch,(xmin,ymin),patch)
   elif kind:d.line(p,fill=col,width=max(1,round(width*S)),joint='curve')
   else:d.polygon(p,fill=col)


def plane(x0,x1,y0,y1,z):return [(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z)]
def floorpoly(x0,x1,z0,z1,y=0):return [(x0,y,z0),(x1,y,z0),(x1,y,z1),(x0,y,z1)]
def hexrgb(c):return tuple(int(c[i:i+2],16) for i in (1,3,5))
def rgbmix(a,b,p):return tuple(round(x+(y-x)*p) for x,y in zip(hexrgb(a),hexrgb(b)))

@lru_cache(maxsize=40)
def background(cam_name,theme):
 im=Image.new('RGB',(W,H),'#171827');sc=Scene(CAMS[cam_name]);cam=CAMS[cam_name]
 colors={'amber':('#171c2b','#664238'),'purple':('#21192f','#4d305d'),'teal':('#102c36','#285859'),'gold':('#241c2b','#784737')}
 dark,light=colors[theme]
 # Backdrop and side walls are flat scenic surfaces in the actual camera space.
 for j in range(55):
  y=j*13
  p=max(0,1-abs(y-290)/490)
  sc.poly(plane(-1300,1300,y,y+14,-450),rgbmix(dark,light,p*.8))
 sc.poly([(-1300,0,-450),(-1300,800,-450),(-1300,800,650),(-1300,0,650)],'#361e2e')
 sc.poly([(1300,0,-450),(1300,0,650),(1300,800,650),(1300,800,-450)],'#361e2e')
 # Thin gold circles and scattered stars replace all literal lyric illustrations.
 for r in [230,243,254]:
  sc.line([(r*math.cos(a),295+r*math.sin(a),-448) for a in np.linspace(0,math.tau,121)],'#886b55',.8)
 for a in np.linspace(0,math.tau,25)[:-1]:
  sc.line([(260*math.cos(a),295+260*math.sin(a),-448),(278*math.cos(a),295+278*math.sin(a),-448)],'#886b55',1)
 for x,y,r,p in base.STARS:
  x=(x-640)*1.45;y=710-y*1.25
  sc.line([(x-r,y,-447),(x+r,y,-447)],'#b7a58b',.8)
  sc.line([(x,y-r,-447),(x,y+r,-447)],'#b7a58b',.8)
 # Wooden stage planks, including boards receding below the high camera.
 for row in range(9):
  z=-450+row*110
  for col in range(20):
   x=-1400+col*150+(75 if row%2 else 0)
   c=rgbmix('#683c33','#88543d',.2+.11*((row+col)%5))
   sc.poly(floorpoly(x,x+148,z,z+108),c)
 # Chorus riser.
 sc.poly(floorpoly(-620,620,-300,-50,12),'#624539')
 sc.poly([(-620,0,-50),(620,0,-50),(620,12,-50),(-620,12,-50)],'#2d2632')
 sc.line([(-620,13,-49),(620,13,-49)],'#cda26a',2)
 # Deep red curtains at the sides of the back wall.
 for side in [-1,1]:
  for i in range(48):
   x=side*(595+i*8)
   val=.5+.4*math.cos(i*.8)
   sc.poly(plane(x,x+8,0,680,-427),rgbmix('#672339','#bc4d51',val))
  sc.line([(side*595,0,-425),(side*595,680,-425)],'#ce9c61',3)
 # Valance scallops: no marquee and no text anywhere in the scenery.
 sc.poly(plane(-1250,1250,555,820,-415),'#79273c')
 for x in range(-1250,1250,240):
  pts=[(x,665,-409),(x+250,665,-409)]+[(x+125+125*math.cos(a),592-85*math.sin(a),-409) for a in np.linspace(0,math.pi,31)]
  sc.poly(pts,'#8d2d42')
  sc.line(pts[2:],'#d0a266',2)
 # Front stage apron and footlights, with their perspective appropriate to each shot.
 sc.poly([(-1350,-26,370),(1350,-26,370),(1350,0,370),(-1350,0,370)],'#382433')
 sc.line([(-1350,1,370),(1350,1,370)],'#daa963',3)
 for x in range(-1220,1230,95):
  pts=[(x+14*math.cos(a),4,355+10*math.sin(a)) for a in np.linspace(0,math.tau,24)]
  sc.poly(pts,'#231f2a')
  pts=[(x+8*math.cos(a),6,355+6*math.sin(a)) for a in np.linspace(0,math.tau,24)]
  sc.poly(pts,'#f8d998')
 sc.paint(im)
 # Soft theater beams sit behind the performers.
 ov=Image.new('RGBA',im.size);d=ImageDraw.Draw(ov)
 for source,end in [((-720,640,-360),(130,0,135)),((720,640,-360),(-130,0,100))]:
  p,z=cam.project([source,(end[0]-120,end[1],end[2]),(end[0]+120,end[1],end[2])])
  if min(z)>0:d.polygon([(round(x*S),round(y*S)) for x,y in p],fill=(255,226,169,14))
 im=Image.alpha_composite(im.convert('RGBA'),ov).convert('RGB')
 return im

class Actor:
 def __init__(self,scene,x,z,scale,t,ident,chorus):
  self.sc=scene;self.x=x;self.z=z;self.scale=scale;self.t=t;self.ident=ident
  self.phase=t*2.27*math.tau+ident*.8
  self.bob=max(0,math.sin(self.phase))*(8 if chorus else 4)
  self.roll=math.sin(self.phase*.5)*(.055 if chorus else .032)
  self.yaw=math.sin(t*.9+ident)*.045
 def tr(self,pts):
  a=np.array(pts,dtype=float)
  x=a[:,0].copy();y=a[:,1].copy()-105;z=a[:,2].copy()
  a[:,0]=x*math.cos(self.roll)-y*math.sin(self.roll)
  a[:,1]=x*math.sin(self.roll)+y*math.cos(self.roll)+105
  x=a[:,0].copy()
  a[:,0]=x*math.cos(self.yaw)+z*math.sin(self.yaw)
  a[:,2]=-x*math.sin(self.yaw)+z*math.cos(self.yaw)
  a*=self.scale;a[:,0]+=self.x;a[:,1]+=self.bob;a[:,2]+=self.z
  return a
 def poly(self,p,c):self.sc.poly(self.tr(p),c)
 def line(self,p,c,w=1):self.sc.line(self.tr(p),c,w)
 def box(self,x0,x1,y0,y1,z0,z1,front='#f18a49',top=True):
  if front is not None:self.poly([(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],front)
  self.poly([(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],'#cc6036')
  self.poly([(x1,y0,z1),(x1,y0,z0),(x1,y1,z0),(x1,y1,z1)],'#b95433')
  if top:self.poly([(x0,y1,z1),(x1,y1,z1),(x1,y1,z0),(x0,y1,z0)],'#ffb46d')
  self.poly([(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],'#d56b3b')
  self.line([(x0+.6,y1-.6,z1+.1),(x1-.6,y1-.6,z1+.1)],'#ffb26f',.8)
 def ellipse(self,rx,rz,y,col):
  self.poly([(rx*math.cos(a),y,rz*math.sin(a)) for a in np.linspace(0,math.tau,49)],col)
 def ring(self,rx,rz,y,col,w=.6):
  self.line([(rx*math.cos(a),y,rz*math.sin(a)) for a in np.linspace(0,math.tau,49)],col,w)
 def cylinder(self,rx,rz,y0,y1,basecolor,shade):
  for i in range(48):
   a=i*math.tau/48;b=(i+1)*math.tau/48
   c=rgbmix(basecolor,shade,.2+.5*(.5+.5*math.cos(a+1)))
   self.poly([(rx*math.cos(a),y0,rz*math.sin(a)),(rx*math.cos(b),y0,rz*math.sin(b)),(rx*math.cos(b),y1,rz*math.sin(b)),(rx*math.cos(a),y1,rz*math.sin(a))],c)
  self.ellipse(rx,rz,y1+.05,basecolor)


@lru_cache(maxsize=4)
def face_texture(squint,mouth_open):
 # Facial details are part of one projected face, preventing depth-order artifacts.
 im=Image.new('RGBA',(660,573),'#f48b4e');d=ImageDraw.Draw(im)
 def pts(p):return [((x+110)*3,(257-y)*3) for x,y in p]
 def line(p,c,w):d.line(pts(p),fill=c,width=round(w*3),joint='curve')
 def poly(p,c):d.polygon(pts(p),fill=c)
 def rect(x0,x1,y0,y1,c):poly([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],c)
 rect(-110,110,250,257,'#f79959');line([(-107,250),(-107,69)],'#ffad6d',.8)
 for xx in [-66,66]:
  if squint:
   sign=1 if xx<0 else -1
   line([(xx-8*sign,224),(xx+6*sign,215),(xx-8*sign,206)],'#201d24',5)
  else:rect(xx-8,xx+8,203,230,'#201d24')
 if mouth_open:rect(-13,13,159,185,'#251d27')
 else:rect(-13,13,170,174,'#251d27')
 left=[(-5,111),(-33,125),(-33,96),(-5,107)]
 right=[(5,111),(33,125),(33,96),(5,107)]
 poly(left,'#7d2940');poly(right,'#8e3044')
 line(left+[left[0]],'#ebc28a',.8);line(right+[right[0]],'#ebc28a',.8)
 rect(-6,6,103,116,'#3a2430')
 return im


def mascot(sc,t,x,z,scale,ident,singing,energy,chorus):
 a=Actor(sc,x,z,scale,t,ident,chorus)
 # Four legs and two rectangular side tabs preserve the reference silhouette.
 for i,x0 in enumerate([-108,-59,33,82]):
  lift=max(0,math.sin(a.phase+(i%2)*math.pi))*(15 if chorus else 6)
  a.box(x0,x0+28,lift,66,-23,28,top=False)
 arm=math.sin(a.phase*.5)*(15 if chorus else 7)
 a.box(-156,-106,137+arm,192+arm,-24,25,'#eb8044')
 a.box(106,156,137-arm,192-arm,-24,25,'#f68c48')
 a.box(-110,110,66,257,-26,30,None)
 squint=math.sin(t*.87+ident*2.1)>.988 or (chorus and math.sin(a.phase*.25)>.94)
 mouth_open=singing and energy>.07 and math.sin(t*math.tau*4.9+ident*.41)>-.22
 sc.texture(a.tr(plane(-110,110,66,257,30.1)),face_texture(squint,mouth_open))
 # Three-dimensional straw boater, including top rings and woven crown.
 a.cylinder(137,70,265,272,'#dab56d','#aa7e40')
 a.ellipse(137,70,272.4,'#e8c780')
 for inset in [3,8,14,20]:a.ring(137-inset,70-inset*.5,272.7,'#b18c4c',.65)
 a.cylinder(84,49,274,277,'#e7c681','#b78b4c')
 a.cylinder(84,49,297,327,'#e7c681','#b78b4c')
 for yy in range(299,325,5):a.ring(84.2,49.2,yy,'#bb9456',.6)
 for ang in np.linspace(0,math.tau,49):
  a.line([(84.3*math.cos(ang),299,49.3*math.sin(ang)),(84.3*math.cos(ang),325,49.3*math.sin(ang))],'#f7df9e',.65)
 a.cylinder(85,50,277,297,'#862d41','#522333')
 a.ring(85.2,50.2,295,'#bd4e58',1.5)
 a.ellipse(84,49,327.5,'#f1d696')
 for inset in range(5,70,9):a.ring(84-inset,49-inset*.54,327.8,'#c5a163',.7)


def microphone(sc):
 a=Actor(sc,140,129,1,0,0,False);a.bob=0;a.roll=0;a.yaw=0
 a.ellipse(27,19,2,'#292732')
 for x0,x1,y0,y1,z0,z1 in [(-2,2,2,211,-2,2),(-13,13,200,240,-7,7)]:
  a.poly([(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],'#c7bfa7')
  a.poly([(x0,y0,z0),(x0,y0,z1),(x0,y1,z1),(x0,y1,z0)],'#827e73')
  a.poly([(x1,y0,z1),(x1,y0,z0),(x1,y1,z0),(x1,y1,z1)],'#918c7b')
  a.poly([(x0,y1,z1),(x1,y1,z1),(x1,y1,z0),(x0,y1,z0)],'#e3dcc3')
 for yy in range(205,238,5):a.line([(-10,yy,9),(10,yy,9)],'#4c4b49',1.3)


def shadows(im,cam,t,actors):
 ov=Image.new('RGBA',im.size);d=ImageDraw.Draw(ov)
 for x,z,s,i in actors:
  p,depth=cam.project([(x+125*s*math.cos(a),.5,z+38*s*math.sin(a)) for a in np.linspace(0,math.tau,40)])
  if min(depth)>0:d.polygon([(round(xx*S),round(yy*S)) for xx,yy in p],fill=(14,15,22,75))
 return Image.alpha_composite(im.convert('RGBA'),ov).convert('RGB')


def render(t,override=None):
 idx=bisect.bisect_right(STARTS,t)-1
 c=CUES[idx] if idx>=0 and t<CUES[idx]['end'] else None
 sec=c['section'] if c else ('Intro' if t<2.154 else 'Outro')
 chorus=('Chorus' in sec and 'Pre' not in sec) or sec=='Outro'
 theme='gold' if chorus else ('purple' if 'Pre' in sec else ('teal' if sec=='Bridge' else 'amber'))
 name=override or SHOTS[bisect.bisect_right(SHOT_STARTS,t)-1][1]
 cam=CAMS[name];im=background(name,theme).copy()
 energy=float(env[min(len(env)-1,int(t*100))])
 singing=c is not None and t<148.65
 if c and c['end']-c['start']>4:
  syllables=max(1,len(base.re.findall(r'[aeiouy]+',c['text'],base.re.I)))
  singing &= t-c['start'] < max(2.3,min(c['end']-c['start'],syllables*.27+1))
 actors=[(-470,-180,.80,1),(-230,-180,.80,2),(230,-180,.80,3),(470,-180,.80,4),(0,75,1.0,0)]
 im=shadows(im,cam,t,actors);sc=Scene(cam)
 for x,z,s,ident in actors:
  sway=math.sin(t*1.1+ident)*7 if chorus else math.sin(t*.6+ident)*2
  mascot(sc,t,x+sway,z,s,ident,singing if ident==0 else singing and chorus,energy,chorus)
 microphone(sc);sc.paint(im)
 if (42.926<t<46.676) or (90<t<96.463) or t>128.218:
  start=42.926 if t<47 else (90 if t<97 else 128.218)
  # Simple falling confetti is decoration, never an illustration or label.
  d=ImageDraw.Draw(im)
  for x,y,v,p,col in base.CONFETTI:
   yy=(y+(t-start)*v)%870-145;xx=x+32*math.sin((t-start)*1.8+p)
   if yy>630:continue
   sz=2+1.5*math.sin(p)**2;aa=(t-start)*2.1+p
   base.poly(d,[(xx+math.cos(aa)*sz,yy+math.sin(aa)*sz),(xx+math.cos(aa+1.57)*sz*.5,yy+math.sin(aa+1.57)*sz*.5),(xx-math.cos(aa)*sz,yy-math.sin(aa)*sz),(xx-math.cos(aa+1.57)*sz*.5,yy-math.sin(aa+1.57)*sz*.5)],col)
 # Final curtain, without any titles or text overlays.
 if t>149.0:
  d=ImageDraw.Draw(im);edge=640*base.ease((t-149)/2.45)
  for side in [0,1]:
   for x in range(round(edge)):
    val=.5+.38*math.cos(x*.12);col=(int(80+59*val),int(21+15*val),int(35+20*val));xx=x if not side else 1280-x
    base.rect(d,(xx,0,xx+1,635),col)
   xx=edge if not side else 1280-edge;base.line(d,[(xx,0),(xx,635)],'#d4ad70',2)
 # Captions remain anchored to the screen across every camera cut.
 d=ImageDraw.Draw(im);base.rect(d,(0,635,1280,720),'#15151f');base.line(d,[(44,635),(1236,635)],'#6e5946',.8)
 if c:
  cap=base.caption_image(idx);im.paste(cap,(0,round(636*S)),cap)
  d=ImageDraw.Draw(im);p=max(0,min(1,(t-c['start'])/(c['end']-c['start'])))
  base.line(d,[(570,710),(710,710)],'#46404a',2);base.line(d,[(570,710),(570+140*p,710)],'#e7bd76',2)
 base.poly(d,[(35,674),(39,679),(35,684),(31,679)],'#bf945c');base.poly(d,[(1245,674),(1249,679),(1245,684),(1241,679)],'#bf945c')
 fade=min(1,t/.35,(DURATION-t)/.7)
 if fade<1:im=Image.blend(Image.new('RGB',im.size,'#090b13'),im,max(0,fade))
 return im


def render_segment(start,end,dest):
 cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-threads','3','-movflags','+faststart',str(dest)]
 p=subprocess.Popen(cmd,stdin=subprocess.PIPE);first=round(start*FPS);last=round(end*FPS);st=time.time()
 try:
  for i in range(first,last):
   p.stdin.write(render(i/FPS).tobytes())
   if (i-first)%240==0:print(f'{dest.name}: {i-first}/{last-first} frames ({time.time()-st:.1f}s)',flush=True)
 finally:p.stdin.close()
 if p.wait()!=0:raise RuntimeError('Encoder failed')
 print(f'COMPLETE {dest.name} {time.time()-st:.1f}s',flush=True)

if __name__=='__main__':
 if sys.argv[1]=='stills':
  sheet=Image.new('RGB',(1440,810),'#171420')
  for i,name in enumerate(CAMERAS):
   im=render(34.3,name);im.save(WORK/f'v2-{name}.jpg',quality=94)
   thumb=im.resize((480,270),Image.Resampling.LANCZOS);sheet.paste(thumb,((i%3)*480,(i//3)*270))
  sheet.save(WORK/'v2-contact.jpg',quality=93)
 elif sys.argv[1]=='segment':render_segment(float(sys.argv[2]),float(sys.argv[3]),Path(sys.argv[4]))
