from pathlib import Path
import torch,torchaudio,numpy as np,json,re,time
from scipy.signal import butter,sosfiltfilt
from scipy.ndimage import uniform_filter1d,maximum_filter1d,median_filter,binary_closing,binary_opening
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work';P.mkdir(exist_ok=True);torch.hub.set_dir(str(P/'model-cache'));torch.set_num_threads(5)
x=np.load(P/'vocals16.npy');sr=16000
cues=json.loads((ROOT/'data/P-doom-Kaboom.lyrics.json').read_text())
bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H;model=bundle.get_model().eval();labels=bundle.get_labels();lookup={c:i for i,c in enumerate(labels)}
def normalize(s):return re.sub(r"[^A-Z' ]",' ',s.upper().replace('’',"'")).split()
sections=[]
for i,c in enumerate(cues):
 if not sections or c['section']!=sections[-1]['section']:sections.append({'section':c['section'],'start':c['start'],'cues':[]})
 sections[-1]['cues'].append((i,c))
allchars=[];allwords=[];newcues=[]
for k,section in enumerate(sections):
 start=max(0,section['start']-.25);end=min(len(x)/sr,sections[k+1]['start']+.10 if k+1<len(sections) else 151.6)

 if section['section']=='Verse 2':start=49.5
 if k==2:end=49.5
 words=[];owners=[]
 for i,c in section['cues']:
  w=normalize(c['text']);w=['PEE' if token=='P' else token for token in w];words+=w;owners += [i]*len(w)
 transcript='|'.join(words);tokens=[lookup[c] for c in transcript]
 clip=torch.from_numpy(x[round(start*sr):round(end*sr)].copy())[None]
 with torch.inference_mode():em,_=model(clip);em=em.log_softmax(-1)
 greedy=torch.unique_consecutive(em[0].argmax(-1));decoded=''.join(labels[i] for i in greedy if i!=0).replace('|',' ')
 path,scores=torchaudio.functional.forced_align(em,torch.tensor([tokens],dtype=torch.int32),blank=0)
 spans=torchaudio.functional.merge_tokens(path[0],scores[0].exp(),blank=0)
 assert len(spans)==len(tokens),(len(spans),len(tokens))
 ratio=clip.shape[-1]/sr/em.shape[1]
 chars=[{'char':labels[s.token],'start':start+s.start*ratio,'end':start+s.end*ratio,'score':float(s.score)} for s in spans]
 allchars+=chars
 wi=0;group=[]
 for q,ch in enumerate(chars+[{'char':'|'}]):
  if ch['char']=='|':
   if group:
    word={'word':words[wi],'cue':owners[wi],'start':group[0]['start'],'end':group[-1]['end'],'confidence':float(np.mean([a['score'] for a in group]))}
    allwords.append(word);wi+=1;group=[]
  else:group.append(ch)
 print(section['section'],round(start,2),round(end,2),decoded,flush=True)
(P/'v3-aligned-characters.json').write_text(json.dumps(allchars,indent=2));(P/'v3-aligned-words.json').write_text(json.dumps(allwords,indent=2))
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
