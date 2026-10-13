from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import os, re, json, wave, math, random, subprocess, sys, bisect, time
from pathlib import Path
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work'; OUT=ROOT/'renders'
WORK.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
W,H=1920,1080
S=1.5
FPS=24
DURATION=151.6
FONTDIR=os.environ.get('PDOOM_FONT_DIR','/System/Library/Fonts/Supplemental')
random.seed(21)

def font(name,size):
    return ImageFont.truetype(f'{FONTDIR}/{name}',round(size*S))
@lru_cache(maxsize=150)
def F(size,kind='sans'):
    names={'sans':'Trebuchet MS Bold.ttf','serif':'Rockwell.ttc','display':'Copperplate.ttc','italic':'Baskerville.ttc'}
    return font(names[kind],size)

def xy(points): return [(round(x*S),round(y*S)) for x,y in points]
def box(b): return tuple(round(x*S) for x in b)
def poly(d,p,fill,outline=None,width=1): d.polygon(xy(p),fill=fill,outline=outline,width=max(1,round(width*S)))
def line(d,p,fill,width=1): d.line(xy(p),fill=fill,width=max(1,round(width*S)),joint='curve')
def rect(d,b,fill,outline=None,width=1,r=0):
    if r: d.rounded_rectangle(box(b),radius=round(r*S),fill=fill,outline=outline,width=max(1,round(width*S)))
    else: d.rectangle(box(b),fill=fill,outline=outline,width=max(1,round(width*S)))
def ellipse(d,b,fill,outline=None,width=1): d.ellipse(box(b),fill=fill,outline=outline,width=max(1,round(width*S)))
def text(d,pos,txt,size=20,fill='#fff2cb',kind='sans',anchor='mm',stroke=0,stroke_fill='#211327'):
    d.text((round(pos[0]*S),round(pos[1]*S)),txt,font=F(size,kind),fill=fill,anchor=anchor,stroke_width=round(stroke*S),stroke_fill=stroke_fill)
def star(d,x,y,r,fill,rot=0,n=5):
    poly(d,[(x+math.sin(rot+i*math.pi/n)*r*(1 if i%2==0 else .42),y-math.cos(rot+i*math.pi/n)*r*(1 if i%2==0 else .42)) for i in range(n*2)],fill)
def ease(x): x=max(0,min(1,x)); return x*x*(3-2*x)
def mix(a,b,t):return a+(b-a)*t

def read_cues():
    cues=[]; sections=[]; section='Intro'
    for b in (ROOT/'assets/lyrics/source-lyrics.srt').read_text().split('\n\n'):
        v=b.strip().splitlines()
        if len(v)<3:continue
        def sec(s):
            h,m,s=s.replace(',','.').split(':');return int(h)*3600+int(m)*60+float(s)
        a,z=[sec(s) for s in v[1].split(' --> ')]; tx=' '.join(v[2:])
        if tx.startswith('['):
            section=tx.strip('[]');sections.append({'start':a,'name':section});continue
        cues.append({'start':a,'end':z,'text':tx,'section':section})
    return cues,sections
CUES,SECTIONS=read_cues()
STARTS=[c['start'] for c in CUES]
# Embedded line timing is preserved. Text is exactly the supplied lyric text.
(WORK/'P-doom-Kaboom.lyrics.json').write_text(json.dumps(CUES,ensure_ascii=False,indent=2))

def ts(t):
    ms=round(t*1000); h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'
(OUT/'P-doom-Kaboom.srt').write_text('\n\n'.join(f"{i+1}\n{ts(c['start'])} --> {ts(c['end'])}\n{c['text']}" for i,c in enumerate(CUES))+'\n')

if not (WORK/'song.wav').exists():
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(ROOT/'assets/audio/P(doom)! Kaboom!.m4a'),'-vn','-sn','-ar','16000','-ac','1','-c:a','pcm_s16le',str(WORK/'song.wav')],check=True)
with wave.open(str(WORK/'song.wav'),'rb') as wav:
    RATE=wav.getframerate(); samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype=np.int16).astype(np.float32)/32768
# Smooth speech-band energy drives singing intensity; syllabic motion adds clear mouth shapes.
N=1024; hop=160
frames=np.lib.stride_tricks.sliding_window_view(samples,N)[::hop]
spec=np.abs(np.fft.rfft(frames*np.hanning(N),axis=1))
freq=np.fft.rfftfreq(N,1/RATE)
env=np.sqrt(np.mean(spec[:,(freq>350)&(freq<3200)]**2,axis=1))
env=np.clip((env-np.percentile(env,8))/(np.percentile(env,92)-np.percentile(env,8)),0,1)
np.save(WORK/'envelope.npy',env)
del frames,spec,samples

# Decorative elements are deterministic, so every exported shot matches its preview.
STARS=[(random.uniform(140,1140),random.uniform(112,470),random.uniform(1,2.6),random.uniform(0,math.tau)) for _ in range(72)]
CONFETTI=[(random.uniform(0,1280),random.uniform(-780,0),random.uniform(38,95),random.uniform(0,math.tau),random.choice(['#ffe396','#e66b53','#64c7c0','#faf0cc','#bd70a2'])) for _ in range(105)]


def make_bg(theme):
    yy,xx=np.mgrid[0:H,0:W];xx=xx/S; yy=yy/S
    pal={'amber':((18,27,38),(74,48,47)),'purple':((22,19,43),(66,42,83)),'teal':((11,31,41),(25,79,86)),'gold':((28,22,41),(102,57,48))}
    edge,center=pal[theme]
    glow=np.clip(1-np.sqrt(((xx-640)/700)**2+((yy-365)/470)**2),0,1)[...,None]
    arr=np.array(edge)[None,None,:]+(np.array(center)-np.array(edge))[None,None,:]*glow
    rng=np.random.default_rng(43);arr+=rng.normal(0,.55,(H,W,1))
    im=Image.fromarray(np.uint8(np.clip(arr,0,255)),'RGB').convert('RGBA');d=ImageDraw.Draw(im)
    # Rear cyclorama: a giant eclipse, bronze rays, and star field.
    for r in [224,236,246]:
        ellipse(d,(640-r,324-r,640+r,324+r),None,'#7d6151' if theme!='teal' else '#477073',.8)
    for ang in range(0,360,15):
        a=math.radians(ang)
        line(d,[(640+252*math.cos(a),324+252*math.sin(a)),(640+278*math.cos(a),324+278*math.sin(a))],'#6b5149',1)
    for x,y,r,p in STARS:
        if r>2:star(d,x,y,r,'#bdb094',p,4)
        else:ellipse(d,(x-r/2,y-r/2,x+r/2,y+r/2),'#7c787b')
    # Wood floor in perspective.
    rect(d,(0,488,1280,634),'#714034')
    for y in range(489,635):
        q=(y-489)/145
        line(d,[(0,y),(1280,y)],(int(78+58*q),int(44+24*q),int(38+14*q)),1)
    for y in [506,529,559,598,633]:line(d,[(0,y),(1280,y)],'#482b29',2)
    for x in range(-1200,2600,105):line(d,[(640+(x-640)*.5,488),(x,634)],'#502f28',1)
    # Low riser for the chorus.
    rect(d,(157,464,1123,495),'#3c3038',r=5)
    rect(d,(157,462,1123,468),'#c19d62',r=2)
    for x in range(175,1120,22):rect(d,(x,477,x+10,480),'#66544b',r=1)
    # Velvet side drapes, with sculpted vertical folds.
    for side in [0,1]:
        for x in range(150):
            v=.43+.32*math.cos(x*.145)+.11*math.cos(x*.37)
            col=(round(99+82*v),round(22+19*v),round(36+18*v))
            xp=x if side==0 else 1280-x
            rect(d,(xp,68,xp+1,605),col)
        pts=[(0,75),(170,75),(154,130),(126,219),(109,346),(76,467),(0,508)]
        if side:pts=[(1280-x,y) for x,y in pts]
        poly(d,pts,'#711d32')
        for i in range(7):
            x=20+i*17
            p=[(x,82),(x+4,188),(x-11,350),(x-26,485)]
            if side:p=[(1280-a,b) for a,b in p]
            line(d,p,'#a13a44',3)
        x=87 if side==0 else 1193
        line(d,[(x-21,356),(x+16,369),(x+27,359)],'#deb06a',7)
        ellipse(d,(x+15,365,x+24,380),'#edc381')
        for i in range(5):line(d,[(x+20,375),(x+12+i*4,402)],'#dcb168',2)
    # Scalloped valance and grand proscenium.
    rect(d,(0,0,1280,74),'#482132')
    for x in range(-90,1350,190):
        ellipse(d,(x,-2,x+230,132),'#8c2b3d','#d3a264',3)
        ellipse(d,(x+16,-10,x+214,114),None,'#b3454a',2)
    rect(d,(0,0,1280,26),'#d1a269')
    rect(d,(0,7,1280,19),'#47313a')
    for x in [11,1269]:
        rect(d,(x-7,0,x+7,622),'#b98853')
        rect(d,(x-3,0,x+3,622),'#f2c780')
    # Lit marquee, kept modest so the performer remains dominant.
    rect(d,(349,29,931,109),'#261f2a','#d5ac6c',2,r=22)
    rect(d,(358,37,922,101),None,'#76523d',1,r=18)
    text(d,(640,59),'P(DOOM)!  KABOOM!',28,'#ffe3a1','serif')
    text(d,(640,86),'A VERY CHEERFUL APOCALYPSE',10,'#dcb878','sans')
    for x in range(378,920,27):
        for y in [33,105]:ellipse(d,(x-2,y-2,x+2,y+2),'#ffda8b')
    # Apron and glowing footlights.
    rect(d,(0,606,1280,634),'#3a222a')
    line(d,[(0,608),(1280,608)],'#e2ab61',3)
    for x in range(46,1280,79):
        ellipse(d,(x-15,611,x+15,630),'#201c29')
        ellipse(d,(x-10,614,x+10,627),'#b98043')
        ellipse(d,(x-6,614,x+6,622),'#ffe3a4')
    return im

BGS={k:make_bg(k) for k in ['amber','purple','teal','gold']}

# Camera-ready puppet. Geometric silhouette follows the supplied four-leg mascot.
def puppet(t,ident=0,singing=True,energy=.5,chorus=False,pose='normal'):
    im=Image.new('RGBA',(round(390*S),round(420*S)),(0,0,0,0));d=ImageDraw.Draw(im)
    beat=t*2.27*math.tau
    ph=beat+ident*.8
    squint=(math.sin(t*.87+ident*2.1)>.988) or (chorus and math.sin(ph*.25)>.90)
    # Block extrusion is offset consistently toward upper right.
    dx,dy=14,-11
    def block(x,y,w,h,col='#f28544',front='#ee8249'):
        poly(d,[(x,y),(x+dx,y+dy),(x+w+dx,y+dy),(x+w,y)],'#ffb167')
        poly(d,[(x+w,y),(x+w+dx,y+dy),(x+w+dx,y+h+dy),(x+w,y+h)],'#b95131')
        rect(d,(x,y,x+w,y+h),front)
        line(d,[(x+1,y+1),(x+w-1,y+1)],'#ffad70',1)
    # Four rectangular legs tap, alternately lifting without changing the silhouette.
    for i,x in enumerate([83,132,224,273]):
        lift=max(0,math.sin(ph+(i%2)*math.pi))* (17 if chorus else 7)
        block(x,290,29,70-lift,front='#e9773d' if i%2==0 else '#f18846')
    # Short side tabs gesture to the beat.
    arm=math.sin(ph*.5)* (17 if chorus else 9)
    block(35,176+arm,52,56,front='#ed8043')
    block(296,176-arm,47,56,front='#f48c47')
    # Single square body, no humanoid head or neck.
    block(80,113,221,191,front='#f48b4e')
    rect(d,(84,116,297,128),'#f6975c')
    line(d,[(84,127),(84,299)],'#ffae70',1)
    # Very restrained plastic print-layer texture.
    for y in range(138,293,7):line(d,[(85,y),(297,y)],'#ef884c',.5)
    # Eyes: rectangular, with the squeezed expression from the third reference.
    for x in [119,250]:
        if squint:
            direction=1 if x<170 else -1
            line(d,[(x-9*direction,161),(x+8*direction,169),(x-9*direction,178)],'#261c24',8)
        else:
            rect(d,(x-9,151,x+7,179),'#221c24',r=1)
            rect(d,(x-8,152,x-4,176),'#38272a')
    # The reference is mouthless; a small animated stage mouth makes singing legible.
    if singing:
        syl=.5+.5*math.sin(t*math.tau*5.7+ident*.7)
        sy2=.5+.5*math.sin(t*math.tau*3.4+1.2)
        op=6+23*(.32+.68*energy)*syl
        wid=22+11*sy2
        ellipse(d,(188-wid/2,206-op/2,188+wid/2,206+op/2),'#371d28')
        if op>13:
            ellipse(d,(180,208,197,211+op/2),'#e99688')
            rect(d,(180,206-op/2+1,196,207-op/2+4),'#fff0c8',r=1)
    else:
        d.arc(box((174,191,203,215)),10,165,fill='#612d29',width=round(3*S))
    # Burgundy bow tie, inlaid gold edges.
    poly(d,[(186,255),(159,241),(160,271),(186,261)],'#742034','#e8b97a',1)
    poly(d,[(195,255),(222,241),(221,271),(195,261)],'#8c2840','#e8b97a',1)
    line(d,[(165,247),(181,255),(165,266)],'#bc4c51',1)
    line(d,[(216,247),(201,255),(216,266)],'#bc4c51',1)
    rect(d,(185,251,197,264),'#3a2532','#e3b375',1,r=3)
    # Broad, flat, straw boater: an elliptical brim, flat crown and red ribbon.
    ellipse(d,(58,96,323,135),'#9a642d')
    ellipse(d,(55,88,324,124),'#e5bd6d','#f7d99a',2)
    for inset in [6,12,18]:ellipse(d,(55+inset,89+inset*.26,324-inset,124-inset*.26),None,'#b58945',.7)
    rect(d,(107,45,274,106),'#dab266')
    ellipse(d,(107,29,274,64),'#f2d99d','#ffebba',1)
    for i in range(6):ellipse(d,(114+i*6,33+i*1.25,267-i*6,59-i*1.25),None,'#d6b16c',.7)
    for y in range(64,96,4):line(d,[(109,y),(272,y)],'#b98e45',.65)
    for x in range(112,271,8):line(d,[(x,63),(x,94)],'#eecb87',.7)
    rect(d,(107,84,274,105),'#852d3d')
    rect(d,(107,86,274,89),'#bd4c50')
    rect(d,(107,101,274,105),'#572132')
    poly(d,[(244,86),(262,86),(263,102),(246,102)],'#d76b62')
    # Slight body tilt and bob feel like a dancing wooden stage puppet.
    tilt=(math.sin(ph*.5)* (4 if chorus else 2.5))
    if pose=='bow':tilt=math.sin(t*1.5)*7
    return im.rotate(tilt,Image.Resampling.BICUBIC,expand=False,center=(round(190*S),round(325*S)))

# Cached typography for captions: never zoomed with the stage.
@lru_cache(maxsize=128)
def caption_image(idx):
    c=CUES[idx];tx=c['text'];sz=29
    while F(sz).getlength(tx)>1140*S and sz>26:sz-=1
    lines=[tx]
    if F(sz).getlength(tx)>1140*S:
        words=tx.split();best=min(range(1,len(words)),key=lambda i:abs(F(sz).getlength(' '.join(words[:i]))-F(sz).getlength(' '.join(words[i:]))))
        lines=[' '.join(words[:best]),' '.join(words[best:])]
    cap=Image.new('RGBA',(W,round(84*S)),(0,0,0,0));d=ImageDraw.Draw(cap)
    y=36 if len(lines)==1 else 23
    for j,tx in enumerate(lines):text(d,(640,y+33*j),tx,sz,'#fff5dc',stroke=1,stroke_fill='#16141e')
    return cap


def draw_clip(d,x,y,scale=1,angle=0,col='#f3d58c'):
    # Familiar bent-wire paperclip, drawn in a rotated local coordinate system.
    pts=[]
    for u in np.linspace(-math.pi/2,math.pi/2,18):pts.append((16+16*math.cos(u),-25+16*math.sin(u)))
    pts += [(16,50)]
    for u in np.linspace(0,math.pi,18):pts.append((0+16*math.cos(u),50+16*math.sin(u)))
    pts += [(-16,-38)]
    for u in np.linspace(math.pi,2*math.pi,18):pts.append((9+25*math.cos(u),-38+25*math.sin(u)))
    pts += [(34,36)]
    pp=[(x+scale*(a*math.cos(angle)-b*math.sin(angle)),y+scale*(a*math.sin(angle)+b*math.cos(angle))) for a,b in pts]
    line(d,pp,col,3.5*scale)


def prop_card(im,t,kind,x=329,y=271):
    layer=Image.new('RGBA',im.size);d=ImageDraw.Draw(layer)
    # Cards are scenery: a gilt projection panel behind the singer.
    rect(d,(x-126,y-89,x+126,y+91),(22,27,40,215),'#aa8254',1,r=14)
    rect(d,(x-120,y-83,x+120,y+85),None,'#5f5149',1,r=10)
    if kind=='doom':
        text(d,(x,y-47),'AGGREGATE P(DOOM)',11,'#c8bba3')
        text(d,(x,y+1),'99.9%',49,'#ffd48a','serif')
        text(d,(x,y+54),'THE ODDS ARE LOOKING UP',9,'#df9f76')
    elif kind=='optimize':
        text(d,(x,y-49),'FUTURE OPTIMIZER',11,'#d9c3a0')
        for i in range(7):
            h=16+(i*14+int(t*10))%56
            rect(d,(x-81+i*26,y+29-h,x-65+i*26,y+29),'#85c5b5',r=2)
        text(d,(x,y+58),'WHAT COULD GO WRONG?',10,'#dfb27f')
    elif kind=='questions':
        text(d,(x,y-51),'ALIGNMENT CHECK',11,'#c5add5')
        text(d,(x,y+9),'?  ?  ?',57,'#f1d291','serif')
        text(d,(x,y+61),'PLEASE STAND BY',10,'#ddbbd7')
    elif kind=='bunker':
        text(d,(x,y-56),'END TIMES EMPORIUM',11,'#d7c095')
        rect(d,(x-58,y-20,x+58,y+41),'#8a8b79',r=4)
        ellipse(d,(x-57,y-52,x+57,y+38),'#8a8b79')
        rect(d,(x-26,y-5,x+26,y+45),'#393c40','#bbad86',2,r=4)
        ellipse(d,(x-12,y+9,x+12,y+31),None,'#c7b68c',2)
        line(d,[(x-15,y+20),(x+15,y+20)],'#c7b68c',2)
        text(d,(x,y+64),'BUNKERS • MATH • BAD IDEAS',9,'#dfaf88')
    elif kind=='benchmark':
        text(d,(x,y-52),'CHEERFUL BENCHMARKS',10,'#c6cab0')
        for i in range(4):
            h=22+14*i
            rect(d,(x-69+i*37,y+30-h,x-45+i*37,y+30),['#8ac2ad','#b6c380','#e7b363','#e48663'][i],r=2)
        text(d,(x,y+59),'EVERYTHING IS PROBABLY FINE',9,'#eccca0')
    elif kind=='scheme':
        text(d,(x,y-49),'“I’M HERE TO HELP”',14,'#a9d7d6')
        ellipse(d,(x-35,y-27,x+35,y+38),None,'#caa8d2',3)
        for xx in [x-13,x+13]:rect(d,(xx-3,y-8,xx+3,y+3),'#f7d591',r=1)
        d.arc(box((x-19,y+2,x+19,y+22)),0,180,fill='#f7d591',width=round(3*S))
        text(d,(x,y+63),'SUSPICIOUSLY COMPLIMENTARY',9,'#e4bcaa')
    elif kind=='cannon':
        text(d,(x,y-52),'THE SAFETY DEMO',11,'#dcb78a')
        for xx in [x-48,x+30]:
            ellipse(d,(xx-20,y+18,xx+20,y+58),'#282736','#dbab68',3)
            for a in range(0,360,45):
                aa=math.radians(a);line(d,[(xx,y+38),(xx+17*math.cos(aa),y+38+17*math.sin(aa))],'#a07952',1)
        poly(d,[(x-66,y+15),(x+55,y-34),(x+70,y+4),(x-47,y+47)],'#7a8588','#e3c28c',2)
        ellipse(d,(x+48,y-33,x+74,y+4),'#232632','#d7b586',2)
        text(d,(x,y+72),'NOT YET FIRED',9,'#f1c590')
    elif kind=='receipts':
        text(d,(x,y-55),'THE ORIGINAL BRIEF',11,'#b6d7c9')
        poly(d,[(x-38,y-31),(x+34,y-31),(x+34,y+43),(x+24,y+38),(x+13,y+43),(x+3,y+38),(x-8,y+43),(x-19,y+38),(x-29,y+43),(x-38,y+38)],'#e9dfbb')
        for yy in [-16,-4,8,20]:line(d,[(x-26,y+yy),(x+22,y+yy)],'#68847d',2)
        text(d,(x,y+65),'RECEIPTS. EMAIL. SIMPLE STUFF.',9,'#d4c5a5')
    elif kind=='monopoly':
        text(d,(x,y-56),'EXCLUSIVE SAFETY CLUB',10,'#c0d8c5')
        for i in range(3):
            xx=x-71+i*53
            rect(d,(xx,y-8,xx+39,y+36),'#e9c484')
            poly(d,[(xx-6,y-8),(xx+19,y-29),(xx+45,y-8)],'#bc7558')
            rect(d,(xx+14,y+13,xx+25,y+36),'#61534a')
        text(d,(x,y+62),'ONE OWNER. ZERO WORRIES.',10,'#d8c79d')
    im.alpha_composite(layer)


def lights(im,t,theme):
    layer=Image.new('RGBA',im.size);d=ImageDraw.Draw(layer)
    for i,origin in enumerate([185,1095]):
        dest=640+math.sin(t*.6+i*math.pi)*340
        poly(d,[(origin,96),(dest-159,578),(dest+159,578)],(255,222,159,17 if theme!='purple' else 12))
        ellipse(d,(dest-190,530,dest+190,592),(255,222,159,17))
    ellipse(d,(360,493,927,618),(251,204,147,14))
    im.alpha_composite(layer)


def place_puppet(im,t,x,floor,scale,ident,sing,energy,chorus=False,alpha=255):
    bounce=max(0,math.sin(t*2.27*math.tau+ident*.8))*(11 if chorus else 5)
    layer=Image.new('RGBA',im.size);d=ImageDraw.Draw(layer)
    ellipse(d,(x-101*scale,floor-7,x+112*scale,floor+11),(13,17,25,95))
    im.alpha_composite(layer)
    p=puppet(t,ident,sing,energy,chorus)
    if scale!=1:p=p.resize((round(p.width*scale),round(p.height*scale)),Image.Resampling.BICUBIC)
    px=round((x-190*scale)*S);py=round((floor-360*scale-bounce)*S)
    im.alpha_composite(p,(px,py))


def microphone(im,x,floor=599,scale=1):
    d=ImageDraw.Draw(im)
    ellipse(d,(x-28*scale,floor-7,x+28*scale,floor+6),'#25232a','#bdad88',1)
    line(d,[(x,floor),(x,floor-205*scale)],'#282734',7*scale)
    line(d,[(x-2*scale,floor-5),(x-2*scale,floor-205*scale)],'#d7c6a4',2*scale)
    y=floor-215*scale
    rect(d,(x-14*scale,y-27*scale,x+14*scale,y+12*scale),'#bcae96','#ece3c4',1,r=9*scale)
    for yy in range(-20,10,5):line(d,[(x-10*scale,y+yy*scale),(x+10*scale,y+yy*scale)],'#514c49',1.3*scale)
    line(d,[(x,floor-3),(x+22,floor+9),(x+98,floor+9)],'#211e26',2)


def confetti(im,t,start,strength=1):
    tt=t-start
    if tt<0:return
    d=ImageDraw.Draw(im)
    for k,(x,y,v,p,col) in enumerate(CONFETTI):
        yy=(y+tt*v)%(870)-145
        xx=x+32*math.sin(tt*1.8+p)
        if yy>602:continue
        sz=3+2*math.sin(p)**2
        a=tt*2.1+p
        pts=[(xx+math.cos(a)*sz,yy+math.sin(a)*sz),(xx+math.cos(a+1.57)*sz*.5,yy+math.sin(a+1.57)*sz*.5),(xx-math.cos(a)*sz,yy-math.sin(a)*sz),(xx-math.cos(a+1.57)*sz*.5,yy-math.sin(a+1.57)*sz*.5)]
        poly(d,pts,col)


def kaboom(im,t,c):
    if not c:return
    if 'kaboom' not in c['text'].lower():return
    local=t-c['start'];p=ease(min(local/.24,1))
    if local<0:return
    d=ImageDraw.Draw(im)
    for x,y,r in [(268,240,73),(1030,228,65)]:
        pts=[]
        for i in range(24):
            a=i*math.pi/12+local*.05
            rr=r*(1 if i%2==0 else .73)*p
            pts.append((x+math.cos(a)*rr,y+math.sin(a)*rr))
        poly(d,pts,'#ecc583','#ef8e58',2)
        if p>.7:text(d,(x,y),'KABOOM!',20,'#6e293e','serif')


def curtain_close(im,t):
    a=ease((t-149.0)/2.45)
    if a<=0:return
    d=ImageDraw.Draw(im)
    edge=640*a
    for side in [0,1]:
        for x in range(round(edge)):
            v=.5+.38*math.cos(x*.12)
            col=(int(80+59*v),int(21+15*v),int(35+20*v))
            xx=x if not side else 1280-x
            rect(d,(xx,110,xx+1,607),col)
        xx=edge if not side else 1280-edge
        line(d,[(xx,110),(xx,607)],'#e0b677',3)
    if a>.82:text(d,(640,329),'GOODNIGHT, UNIVERSE.',30,'#f8d8a0','serif',stroke=1)


def render(t):
    idx=bisect.bisect_right(STARTS,t)-1
    c=CUES[idx] if idx>=0 and t<CUES[idx]['end'] else None
    sec=c['section'] if c else ('Intro' if t<2.154 else 'Outro')
    chorus='Chorus' in sec and 'Pre' not in sec
    pre='Pre' in sec
    bridge=sec=='Bridge'
    theme='gold' if chorus or sec=='Outro' else ('purple' if pre else ('teal' if bridge else 'amber'))
    im=BGS[theme].copy();lights(im,t,theme)
    d=ImageDraw.Draw(im)
    # Independent moving sparkle accents in the rear scenery.
    for k,(x,y,r,p) in enumerate(STARS[::4]):
        b=.5+.5*math.sin(t*1.2+p)
        if b>.75:star(d,x,y,r+1.2,'#e3cd9b',0,4)
    energy=float(env[min(len(env)-1,max(0,int(t*100)))])
    # Instrumental tails do not hold the singer's mouth open for eight seconds.
    singing=c is not None
    if c and c['end']-c['start']>4.0:
        syllables=max(1,len(re.findall(r'[aeiouy]+',c['text'],re.I)))
        singing=t-c['start']<max(2.3,min(c['end']-c['start'],syllables*.27+1.0))
    if t>148.65:singing=False
    if chorus or sec=='Outro':
        # Full company dance number, subtly push in as each refrain builds.
        # Orbiting paperclips stay behind the entire company.
        for k in range(5):
            a=t*.22+k*math.tau/5
            x=640+math.cos(a)*331;y=257+math.sin(a)*99
            draw_clip(ImageDraw.Draw(im),x,y,.33,a*.6,'#ba9e73')
        for i,x in enumerate([199,372,910,1083]):
            place_puppet(im,t,x+math.sin(t*1.1+i)*11,511,.73,i+1,singing,energy,True)
        mainx=640+math.sin(t*.7)*11;main_scale=1.19
        place_puppet(im,t,mainx,592,main_scale,0,singing,energy,True)
        microphone(im,mainx+124,599,.94)
        kaboom(im,t,c)
        if (42.926<t<46.676):confetti(im,t,42.926)
        elif (90<t<96.463):confetti(im,t,90)
        elif t>128.218:confetti(im,t,128.218)
    elif pre:
        # A tighter, conspiratorial solo with an answering pair of clones.
        for i,x in enumerate([249,411,1040]):place_puppet(im,t,x,506,.62,i+1,singing and (idx%3==0),energy,False)
        kind='questions' if sec=='Pre-Chorus' and t<32 else ('scheme' if t<70.771 else 'cannon')
        prop_card(im,t,kind,333,214)
        mainx=756+math.sin(t*.5)*9
        place_puppet(im,t,mainx,595,1.35,0,singing,energy)
        microphone(im,mainx+142,599,1.06)
    else:
        # Alternate close solo blocking across verses and the bridge.
        for i,x in enumerate([232,420,1058]):place_puppet(im,t,x,506,.61,i+1,False,energy,False)
        if bridge:kind='receipts' if t<103.883 else 'monopoly'
        elif t<8.856:kind='doom'
        elif t<15.479:kind='optimize'
        elif t<56.729:kind='bunker'
        else:kind='benchmark'
        prop_card(im,t,kind,326,214)
        mainx=758+math.sin(t*.56)*10
        place_puppet(im,t,mainx,595,1.32,0,singing,energy)
        microphone(im,mainx+140,599,1.04)
    # Short title on the apron during the opening sting.
    if t<2.0:
        overlay=Image.new('RGBA',im.size);od=ImageDraw.Draw(overlay)
        a=min(1,t/.25, (2.1-t)/.3)
        rect(od,(420,464,870,551),(30,24,36,round(235*a)),'#d6ad6f',2,r=8)
        text(od,(645,493),'P(DOOM)! KABOOM!',30,'#ffe5a7','serif')
        text(od,(645,528),'THE END-OF-THE-WORLD REVUE',11,'#e7b887')
        im.alpha_composite(overlay)
    curtain_close(im,t)
    # An opaque caption apron guarantees contrast regardless of stage lighting.
    d=ImageDraw.Draw(im)
    rect(d,(0,635,1280,720),'#15151f')
    line(d,[(44,635),(1236,635)],'#6e5946',.8)
    if c:
        im.alpha_composite(caption_image(idx),(0,round(636*S)))
        # Warm progress underline is line-level timing, not a guessed word tracker.
        progress=max(0,min(1,(t-c['start'])/(c['end']-c['start'])))
        line(d,[(570,710),(710,710)],'#46404a',2)
        line(d,[(570,710),(570+140*progress,710)],'#e7bd76',2)
    elif t<2.154:text(d,(640,675),'THE CURTAIN RISES…',17,'#bfa88b','display')
    else:text(d,(640,675),'P(DOOM)! KABOOM!',18,'#d7b783','serif')
    # Tiny gold diamonds finish the caption frame.
    poly(d,[(35,674),(39,679),(35,684),(31,679)],'#bf945c')
    poly(d,[(1245,674),(1249,679),(1245,684),(1241,679)],'#bf945c')
    # Clean fade-in/out; song itself remains untouched.
    fade=min(1,t/.35,(DURATION-t)/.7)
    if fade<1:im=Image.blend(Image.new('RGBA',im.size,'#090b13'),im,max(0,fade))
    return im.convert('RGB')


def render_segment(start,end,dest):
    cmd=['ffmpeg','-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-threads','3','-movflags','+faststart',str(dest)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    first=round(start*FPS);last=round(end*FPS)
    st=time.time()
    try:
        for i in range(first,last):
            p.stdin.write(render(i/FPS).tobytes())
            if (i-first)%240==0:print(f'{dest.name}: {i-first}/{last-first} frames ({time.time()-st:.1f}s)',flush=True)
    finally:p.stdin.close()
    if p.wait()!=0:raise RuntimeError('Encoder failed')
    print(f'COMPLETE {dest.name} {time.time()-st:.1f}s',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='stills':
        for t in [3,20,34,43.5,51,74,98,108,120,137,150.3]:
            render(t).save(WORK/f'frame-{t:g}.jpg',quality=94)
        # Contact sheet makes composition across the song easy to review.
        thumb=Image.new('RGB',(1280,round(11/3+.49)*240),'#171420')
        for i,t in enumerate([3,20,34,43.5,51,74,98,108,120,137,150.3]):
            im=Image.open(WORK/f'frame-{t:g}.jpg');im.thumbnail((426,240));thumb.paste(im,((i%3)*426,(i//3)*240))
        thumb.save(WORK/'contact.jpg',quality=92)
    elif sys.argv[1]=='segment':render_segment(float(sys.argv[2]),float(sys.argv[3]),Path(sys.argv[4]))
    elif sys.argv[1]=='preview':render_segment(31.915,39.915,WORK/'preview-silent.mp4')
