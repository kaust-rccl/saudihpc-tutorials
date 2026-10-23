import os, sys, io, re, json, time, math, random, zipfile, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
warnings.filterwarnings('ignore'); torch.set_num_threads(12)
SP=str(Path(__file__).resolve().parent)+'/'; C=Path(os.environ.get('CROPGUARD_CACHE','notebooks/data/cache'))
exec(open(SP+'synth_snippet.py').read()); exec(open(SP+'regimes_snippet.py').read())
z=np.load(C/'pv_200.npz',allow_pickle=True); pv={k:z[k] for k in z.files}; pv['mask']=np.unpackbits(pv.pop('mask_packed'),axis=-1).astype(bool)
zr=np.load(C/'rocole.npz',allow_pickle=True); ro={k:zr[k] for k in zr.files}
Xc=np.load(C/'feat_pv_clean.npy'); Xr=np.load(C/'feat_rocole.npy'); SEED=42
classes=sorted(set(pv['cls'])); y=np.array([classes.index(c) for c in pv['cls']]); is_h=np.array([c.endswith('healthy') for c in classes])
trval,test_i=next(GroupShuffleSplit(1,test_size=.2,random_state=SEED).split(Xc,y,pv['gid']))
tr_rel,va_rel=next(GroupShuffleSplit(1,test_size=.25,random_state=SEED).split(Xc[trval],y[trval],pv['gid'][trval]))
train_i,val_i=np.random.RandomState(SEED).permutation(trval[tr_rel]),trval[va_rel]
mu,sd=Xc[train_i].mean(0),Xc[train_i].std(0)+1e-6
# RoCoLe split by plant: dev half chooses the regime, test half is locked
up=np.unique(ro['plant']); rp=np.random.RandomState(123).permutation(up); dev_pl=set(rp[:len(up)//2]); dev=np.array([p in dev_pl for p in ro['plant']]); sick=ro['sick'].astype(int)
print('RoCoLe dev images',dev.sum(),'test images',(~dev).sum(),'| sick share dev %.2f test %.2f'%(sick[dev].mean(),sick[~dev].mean()),flush=True)
net=torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); net.fc=nn.Identity(); net.eval()
MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1); STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)
@torch.no_grad()
def extract(imgs,bs=128): return np.concatenate([net(((torch.from_numpy(np.ascontiguousarray(imgs[i:i+bs])).permute(0,3,1,2).float()/255-MEAN)/STD)).numpy() for i in range(0,len(imgs),bs)])
def regime_features(name, seed_base):
    f=C/f'feat_reg_{name}.npy'
    if f.exists(): return np.load(f)
    if name=='synth_0.5': return np.load(C/'feat_aug_train_s05.npy')
    if name=='synth_1.0': return np.load(C/'feat_aug_train_s10.npy')
    t=time.time(); imgs=np.stack([make_regime(name,int(i),train_i,seed_base+int(i)) for i in train_i]); X=extract(imgs); np.save(f,X); print(f'  built {name}: {time.time()-t:.0f}s',flush=True); return X
class Head(nn.Module):
    def __init__(s,h,d,n): super().__init__(); s.net=nn.Linear(512,n) if h==0 else nn.Sequential(nn.Linear(512,h),nn.ReLU(),nn.Dropout(d),nn.Linear(h,n))
    def forward(s,x): return s.net(x)
CFG=dict(hidden=512,dropout=.5,lr=1e-3,wd=1e-5,ls=.1)
def fit(X,yy,binary,seed,epochs=30):
    torch.manual_seed(seed); yy=(~is_h[yy]).astype(int) if binary else yy; X=torch.tensor((X-mu)/sd,dtype=torch.float32); yy=torch.tensor(yy)
    h=Head(CFG['hidden'],CFG['dropout'],2 if binary else len(classes)); opt=torch.optim.Adam(h.parameters(),lr=CFG['lr'],weight_decay=CFG['wd']); lf=nn.CrossEntropyLoss(label_smoothing=CFG['ls'])
    for _ in range(epochs):
        h.train(); o=torch.randperm(len(X))
        for i in range(0,len(o),256): j=o[i:i+256]; opt.zero_grad(); lf(h(X[j]),yy[j]).backward(); opt.step()
    return h.eval()
def probs(h,X):
    with torch.no_grad(): return torch.softmax(h(torch.tensor((X-mu)/sd,dtype=torch.float32)),1).numpy()
def score(P,binary): return P[:,1] if binary else 1-P[:,is_h].sum(1)
def evaluate(Xa, which):                       # which: boolean mask over RoCoLe images
    X=np.vstack([Xc[train_i],Xa]) if Xa is not None else Xc[train_i]; yy=np.concatenate([y[train_i],y[train_i]]) if Xa is not None else y[train_i]
    out={}
    for binary in (False,True):
        a=[];b=[];c=[]
        for s in range(3):
            h=fit(X,yy,binary,SEED+10*s); sc=score(probs(h,Xr[which]),binary); a.append(roc_auc_score(sick[which],sc)); b.append(balanced_accuracy_score(sick[which],sc>=.5))
            if not binary: c.append(accuracy_score(y[val_i],probs(h,Xc[val_i]).argmax(1)))
        out['bin' if binary else 'der']=(np.mean(a),np.mean(b)); 
        if not binary: out['pv_acc']=np.mean(c)
    return out
if __name__=='__main__':
    regimes=['none','synth_0.2','synth_0.35','synth_0.5','synth_1.0','photo','rrc','bgswap','bgswap+rrc','synth_0.35+rrc']
    rows={}
    for k,name in enumerate(regimes):
        Xa=None if name=='none' else regime_features(name,500_000+k*100_000)
        r=evaluate(Xa,dev); rows[name]=r; print(f"{name:16s} PV val acc {r['pv_acc']:.3f} | DEV derived AUC {r['der'][0]:.3f} bal {r['der'][1]:.3f} | DEV direct AUC {r['bin'][0]:.3f} bal {r['bin'][1]:.3f}",flush=True)
    pd.DataFrame({n:{'pv_val_acc':r['pv_acc'],'dev_derived_auc':r['der'][0],'dev_derived_bal':r['der'][1],'dev_direct_auc':r['bin'][0],'dev_direct_bal':r['bin'][1]} for n,r in rows.items()}).T.to_pickle(SP+'sweep1.pkl')
