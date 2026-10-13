from pathlib import Path
import torch,torchaudio,subprocess,wave,numpy as np,time
from demucs.pretrained import get_model
from demucs.apply import apply_model
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'work';P.mkdir(exist_ok=True)
torch.hub.set_dir(str(P/'model-cache'));torch.set_num_threads(6)
print('Downloading separator',flush=True)
model=get_model('htdemucs').cpu().eval()
print('Downloading alignment model',flush=True)
bundle=torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
align=bundle.get_model().cpu().eval()
print('Models ready. Separating vocals locally.',flush=True)
subprocess.run(['ffmpeg','-y','-v','error','-i',str(ROOT/'assets/audio/P(doom)! Kaboom!.m4a'),'-vn','-sn','-ar','44100','-ac','2',str(P/'mix441.wav')],check=True)
with wave.open(str(P/'mix441.wav')) as f:
 data=np.frombuffer(f.readframes(f.getnframes()),np.int16).reshape(-1,2).T.copy().astype(np.float32)/32768
mix=torch.from_numpy(data)
ref=mix.mean(0);mean=ref.mean();std=ref.std();norm=(mix-mean)/std
st=time.time()
with torch.inference_mode():out=apply_model(model,norm[None],device='cpu',shifts=0,split=True,overlap=.1,progress=True,num_workers=0)[0]
out=out*std+mean
vocals=out[model.sources.index('vocals')]
np.save(P/'vocals441.npy',vocals.numpy())
mono=torchaudio.functional.resample(vocals.mean(0),44100,16000).numpy()
np.save(P/'vocals16.npy',mono)
with wave.open(str(P/'vocals16.wav'),'wb') as w:
 w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes((np.clip(mono,-1,1)*32767).astype(np.int16).tobytes())
print('Vocal extraction complete',round(time.time()-st,1),'seconds',flush=True)
