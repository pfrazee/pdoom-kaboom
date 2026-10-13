from PIL import Image,ImageDraw,ImageFont
import create_video_v3 as v
shots=[(1.77,'Slide-in entrance'),(9.5,'Direct-to-camera lean'),(12.6,'Solo spotlight'),(15.38,'Pop-up entrance'),(20.3,'Call and response'),(33.0,'Chorus kicks'),(37.2,'Three-panel chorus'),(48.4,'Passing parade'),(54.0,'Hat tip'),(76.2,'Close-up aside'),(129.1,'Arms-up finale'),(148.3,'Company bow')]
sheet=Image.new('RGB',(1440,1200),'#171420');f=ImageFont.truetype('/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf',17)
for i,(t,label) in enumerate(shots):
 im=v.render(t);x=i%3*480;y=i//3*300
 sheet.paste(im.resize((480,270),Image.Resampling.LANCZOS),(x,y));ImageDraw.Draw(sheet).text((x+14,y+276),f'{i+1:02}  {label}',fill='#f1d296',font=f)
sheet.save(v.OUT/'P-doom-Kaboom-v3-scenes.jpg',quality=95)
v.render(3.05).save(v.OUT/'P-doom-Kaboom-v3-poster.jpg',quality=95)
print('Saved twelve-scene overview and poster.')
