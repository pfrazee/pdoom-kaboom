from pathlib import Path
import numpy as np,json
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import uniform_filter1d,maximum_filter1d,binary_closing,binary_opening
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work';P.mkdir(exist_ok=True);x=np.load(P/'vocals16.npy');sr=16000
cues=json.loads((ROOT/'data/P-doom-Kaboom.lyrics.json').read_text());allwords=json.loads((P/'v3-aligned-words.json').read_text());allchars=json.loads((P/'v3-aligned-characters.json').read_text());newcues=[]
# Derive the binary mouth track from the separated voice, constrained by acoustic word alignment.
sos=butter(3,[150,3100],btype='bandpass',fs=sr,output='sos')
y=sosfiltfilt(sos,x);power=uniform_filter1d(y*y,size=round(sr*.024))
times=np.arange(int(np.ceil(len(x)/sr*100)))/100
rms=np.sqrt(power[np.minimum(len(power)-1,(times*sr).astype(int))]);local=maximum_filter1d(rms,size=43)
threshold=np.maximum(np.percentile(rms,90)*.07,local*.24)
active=rms>threshold
# Phrase ranges suppress accompaniment bleed; the vocal envelope controls breaths and sustained notes.
wordgate=np.zeros(len(times),bool)
for cue_id in sorted(set(w['cue'] for w in allwords)):
 ws=[w for w in allwords if w['cue']==cue_id]
 a=max(0,round((ws[0]['start']-.07)*100));b=min(len(times),round((ws[-1]['end']+.065)*100));wordgate[a:b]=True
active &= wordgate
# Preserve short syllable closures, while removing sub-frame chatter.
active=binary_closing(active,structure=np.ones(3));active=binary_opening(active,structure=np.ones(4))
# Bilabial consonants need a closed mouth even when the voice is still audible.
for j,c in enumerate(allchars):
 if c['char'] not in 'BMP' or c['score']<.25:continue
 center=(c['start']+c['end'])/2
 lo=max(c['start']-.025,center-.045);hi=min(c['end']+.025,center+.045)
 if hi-lo<.060:lo=center-.035;hi=center+.035
 active[max(0,round(lo*100)):min(len(active),round(hi*100))]=False
# Short acoustic articulations separate syllables and adjacent sung words.
# These closures follow aligned consonants/word boundaries, never a repeating timer.
for c in allchars:
 if c['char'] in 'TDK' and c['score']>.6:
  mid=(c['start']+c['end'])/2
  active[max(0,round((mid-.025)*100)):min(len(active),round((mid+.025)*100))]=False
 elif c['char']=='|' and c['score']>.6 and c['end']-c['start']>=.03:
  mid=(c['start']+c['end'])/2
  active[max(0,round((mid-.022)*100)):min(len(active),round((mid+.022)*100))]=False

# Export a dense track to avoid interpolation or a periodic mouth oscillator.
np.savez(P/'v3-mouth-track.npz',times=times,open=active.astype(np.uint8),energy=np.clip(rms/(np.percentile(rms,92)+1e-9),0,1),rms=rms,threshold=threshold,wordgate=wordgate.astype(np.uint8))
# Caption text remains exactly supplied; cue onsets now follow the actual sung words.
for i,c in enumerate(cues):
 ws=[w for w in allwords if w['cue']==i]
 a=max(0,ws[0]['start']-.08);b=ws[-1]['end']+.15
 newcues.append({**c,'start':a,'end':b})
for i,c in enumerate(newcues):
 if i+1<len(newcues):c['end']=min(c['end'],newcues[i+1]['start'])
(P/'v3-cues.json').write_text(json.dumps(newcues,indent=2,ensure_ascii=False))
intervals=[];begin=None
for i,v in enumerate(active):
 if v and begin is None:begin=i/100
 if not v and begin is not None:intervals.append([begin,i/100]);begin=None
if begin is not None:intervals.append([begin,len(active)/100])
(P/'v3-mouth-open-intervals.json').write_text(json.dumps(intervals,indent=2))
print('Aligned',len(allwords),'words;',len(intervals),'vocal mouth openings;',round(active.mean()*100,1),'percent open',flush=True)
