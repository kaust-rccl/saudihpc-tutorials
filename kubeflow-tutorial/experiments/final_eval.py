import sys; sys.argv=['x']
src=open(str(__import__('pathlib').Path(__file__).resolve().parent/'sweep2.py')).read().split("if __name__=='__main__':")[0]
exec(src)
test=~dev; plants=ro['plant']; N_SEEDS=5
cands={'clean only (no augmentation)':None,'earlier regime: synthetic field 0.5':regime_features('synth_0.5',0),'chosen: flip+rrc0.1+photo1':regime_features('flip+rrc0.1+photo1',0)}
res={}
for name,Xa in cands.items():
    X=np.vstack([Xc[train_i],Xa]) if Xa is not None else Xc[train_i]; yy=np.concatenate([y[train_i]]*(2 if Xa is not None else 1))
    out={'pv_clean_test_acc':[], 'der':[], 'bin':[]}
    for s in range(N_SEEDS):
        hd=fit(X,yy,False,SEED+10*s); hb=fit(X,yy,True,SEED+10*s)
        out['pv_clean_test_acc'].append(accuracy_score(y[test_i],probs(hd,Xc[test_i]).argmax(1)))
        out['der'].append(score(probs(hd,Xr),False)); out['bin'].append(score(probs(hb,Xr),True))
    res[name]={'acc':np.mean(out['pv_clean_test_acc']),'der':np.mean(out['der'],axis=0),'bin':np.mean(out['bin'],axis=0)}
def m(sc,idx): return roc_auc_score(sick[idx],sc[idx]),balanced_accuracy_score(sick[idx],sc[idx]>=.5)
tidx=np.where(test)[0]
print('LOCKED RoCoLe half: %d images, %d plants (never used to choose anything)\n'%(test.sum(),len(set(plants[test]))))
print(f"{'regime':40s} {'PV clean test acc':>17s} | {'derived AUC':>11s} {'bal@.5':>7s} | {'direct AUC':>10s} {'bal@.5':>7s}")
for n,r in res.items():
    d=m(r['der'],tidx); b=m(r['bin'],tidx); print(f"{n:40s} {r['acc']:17.3f} | {d[0]:11.3f} {d[1]:7.3f} | {b[0]:10.3f} {b[1]:7.3f}")
rng=np.random.RandomState(0); up_=np.unique(plants[test]); by={p:np.where((plants==p)&test)[0] for p in up_}
def boot(a,b,key,which):
    ds=[]
    for _ in range(1000):
        idx=np.concatenate([by[p] for p in rng.choice(up_,len(up_))]); ds.append(m(res[a][key],idx)[which]-m(res[b][key],idx)[which])
    return np.mean(ds),np.percentile(ds,2.5),np.percentile(ds,97.5)
base='clean only (no augmentation)'
for other in ('earlier regime: synthetic field 0.5','chosen: flip+rrc0.1+photo1'):
    print(f'\n{other}  minus  clean only  (bootstrap over locked plants):')
    for key,lab in (('der','derived AUC'),('der','derived bal acc'),('bin','direct AUC'),('bin','direct bal acc')):
        w=1 if 'bal' in lab else 0; mm,lo,hi=boot(other,base,key,w); print(f'   {lab:18s} {mm:+.3f}  (95% CI {lo:+.3f} to {hi:+.3f})')
print('\nshare predicted sick (derived, thr .5):',{n:round(float((r['der'][tidx]>=.5).mean()),2) for n,r in res.items()},'| true share %.2f'%sick[tidx].mean())
