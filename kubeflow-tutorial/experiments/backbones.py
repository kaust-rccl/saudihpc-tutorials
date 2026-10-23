import os, io, sys, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
warnings.filterwarnings('ignore'); torch.set_num_threads(12)
SP=str(Path(__file__).resolve().parent)+'/'; C=Path(os.environ.get('CROPGUARD_CACHE','notebooks/data/cache'))
exec(open(SP+'synth_snippet.py').read()); exec(open(SP+'regimes_snippet.py').read())
z=np.load(C/'pv_200.npz',allow_pickle=True); pv={k:z[k] for k in z.files}; pv['mask']=np.unpackbits(pv.pop('mask_packed'),axis=-1).astype(bool)
zr=np.load(C/'rocole.npz',allow_pickle=True); ro={k:zr[k] for k in zr.files}; SEED=42
classes=sorted(set(pv['cls'])); y=np.array([classes.index(c) for c in pv['cls']]); is_h=np.array([c.endswith('healthy') for c in classes]); N=len(classes)
Xdummy=np.zeros((len(y),1)); trval,test_i=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED).split(Xdummy,y,pv['gid']))
tr_rel,va_rel=next(GroupShuffleSplit(1,test_size=.25,random_state=SEED).split(Xdummy[trval],y[trval],pv['gid'][trval]))
train_i,val_i=np.random.RandomState(SEED).permutation(trval[tr_rel]),trval[va_rel]
up=np.unique(ro['plant']); rp=np.random.RandomState(123).permutation(up); dev_pl=set(rp[:len(up)//2]); dev=np.array([p in dev_pl for p in ro['plant']]); sick=ro['sick'].astype(int)
RECIPE='flip+rrc0.1+photo1'
t=time.time(); aug_imgs=np.stack([make_regime(RECIPE,int(i),train_i,777_000+int(i)) for i in train_i]); print(f'recipe images generated: {time.time()-t:.0f}s',flush=True)
MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1); STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)
def build(name):
    tv=torchvision.models
    if name=='resnet18': m=tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1); m.fc=nn.Identity()
    elif name=='resnet50': m=tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2); m.fc=nn.Identity()
    elif name=='efficientnet_b0': m=tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier=nn.Identity()
    elif name=='convnext_tiny': m=tv.convnext_tiny(weights=tv.ConvNeXt_Tiny_Weights.IMAGENET1K_V1); m.classifier[2]=nn.Identity()
    elif name=='dinov2_vits14': m=torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
    return m.eval()
@torch.no_grad()
def extract(m,imgs,bs=64): return np.concatenate([m(((torch.from_numpy(np.ascontiguousarray(imgs[i:i+bs])).permute(0,3,1,2).float()/255-MEAN)/STD)).numpy() for i in range(0,len(imgs),bs)])
def feats(name):
    out={}
    m=None
    for tag,imgs in (('clean',pv['img']),('aug',aug_imgs),('rocole',ro['img'])):
        f=C/f'feat_bb_{name}_{tag}.npy'
        if f.exists(): out[tag]=np.load(f); continue
        if m is None: m=build(name)
        t=time.time(); out[tag]=extract(m,imgs); np.save(f,out[tag]); print(f'   {name} {tag}: {len(imgs)} imgs in {time.time()-t:.0f}s',flush=True)
    return out
class Head(nn.Module):
    def __init__(s,d,n): super().__init__(); s.net=nn.Linear(d,n)
    def forward(s,x): return s.net(x)
def run(F,after,binary,seed):
    Xc,Xa,Xr=F['clean'],F['aug'],F['rocole']; mu,sd=Xc[train_i].mean(0),Xc[train_i].std(0)+1e-6
    X=np.vstack([Xc[train_i],Xa]) if after else Xc[train_i]; yy=np.concatenate([y[train_i]]*(2 if after else 1)); yy=(~is_h[yy]).astype(int) if binary else yy
    torch.manual_seed(seed); Xt=torch.tensor((X-mu)/sd,dtype=torch.float32); yt=torch.tensor(yy); h=Head(X.shape[1],2 if binary else N); opt=torch.optim.Adam(h.parameters(),lr=1e-3,weight_decay=1e-5); lf=nn.CrossEntropyLoss()
    for _ in range(30):
        h.train(); o=torch.randperm(len(Xt))
        for i in range(0,len(o),256): j=o[i:i+256]; opt.zero_grad(); lf(h(Xt[j]),yt[j]).backward(); opt.step()
    h.eval()
    def P(Z):
        with torch.no_grad(): return torch.softmax(h(torch.tensor((Z-mu)/sd,dtype=torch.float32)),1).numpy()
    pr=P(Xr); s=pr[:,1] if binary else 1-pr[:,is_h].sum(1)
    return roc_auc_score(sick[dev],s[dev]), balanced_accuracy_score(sick[dev],s[dev]>=.5), (None if binary else accuracy_score(y[val_i],P(Xc[val_i]).argmax(1)))
rows={}
for name in sys.argv[1:]:
    t0=time.time()
    try: F=feats(name)
    except Exception as e: print('FAILED',name,repr(e)[:150],flush=True); continue
    r={'dim':F['clean'].shape[1],'extract_s':round(time.time()-t0)}
    for after in (False,True):
        d=[run(F,after,False,SEED+10*s) for s in range(3)]; b=[run(F,after,True,SEED+10*s) for s in range(3)]; k='after' if after else 'before'
        r[f'der_auc_{k}']=np.mean([x[0] for x in d]); r[f'dir_auc_{k}']=np.mean([x[0] for x in b]); r[f'pv_acc_{k}']=np.mean([x[2] for x in d]); r[f'der_bal_{k}']=np.mean([x[1] for x in d])
    rows[name]=r; print(name,{k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()},flush=True)
pd.DataFrame(rows).T.to_pickle(SP+'backbones.pkl')
