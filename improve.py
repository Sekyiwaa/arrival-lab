"""Validation-selected TabPFN experiment. Never overwrites original results."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from benchmark import prepare,haversine,FEATURES
EXTRA=['traveled_km','mean_speed_so_far','speed_20_points','stop_fraction_20','distance_change_20','current_speed_eta','hour_sin','hour_cos']

def history(g,destination):
    g=g.sort_values('time').drop_duplicates('time').copy();g['time']=pd.to_datetime(g.time)
    dt=g.time.diff().dt.total_seconds()/3600
    seg=haversine(g.latitude.shift(),g.longitude.shift(),g.latitude,g.longitude).fillna(0)
    speed=(seg/dt).replace([np.inf,-np.inf],np.nan).clip(0,120).fillna(0)
    elapsed=(g.time-g.time.iloc[0]).dt.total_seconds()/60
    traveled=seg.cumsum();dist=haversine(g.latitude,g.longitude,*destination)
    result=pd.DataFrame({'elapsed_min':elapsed,'traveled_km':traveled,'mean_speed_so_far':traveled/(elapsed/60).replace(0,np.nan),'speed_20_points':speed.rolling(20,min_periods=1).mean(),'stop_fraction_20':speed.lt(3).rolling(20,min_periods=1).mean(),'distance_change_20':dist.diff(20).fillna(0),'current_speed_eta':dist/speed.rolling(20,min_periods=1).mean().clip(lower=3)*60,'hour_sin':np.sin(2*np.pi*g.time.dt.hour/24),'hour_cos':np.cos(2*np.pi*g.time.dt.hour/24)})
    return result.fillna(0)

def enriched():
    f,trips=prepare();pts=pd.read_csv('data/go_track_trackspoints.csv');extras={}
    for tid,g in pts[pts.track_id.isin(f.trip_id.unique())].groupby('track_id'):
        sub=f[f.trip_id.eq(tid)];dest=(sub.destination_lat.iloc[0],sub.destination_lon.iloc[0]);h=history(g,dest)
        for _,r in h.iterrows():extras[(int(tid),round(r.elapsed_min,6))]=r[EXTRA].to_dict()
    add=pd.DataFrame([extras[(int(r.trip_id),round(r.elapsed_min,6))] for _,r in f.iterrows()],index=f.index)
    return pd.concat([f,add],axis=1),trips

def scores(y,p,groups):
    e=np.abs(np.asarray(y)-np.asarray(p));per=pd.DataFrame({'trip':groups,'error':e}).groupby('trip').error.mean()
    return {'mae_min':float(e.mean()),'trip_balanced_mae_min':float(per.mean()),'within_2_min':float((e<=2).mean())}

def run():
    ap=argparse.ArgumentParser();ap.add_argument('--tabpfn',action='store_true');ap.add_argument('--variants',nargs='+',choices=['plus','fast','thinking'],default=['plus','fast']);args=ap.parse_args()
    f,trips=enriched();outer,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(f,groups=f.trip_id))
    ia,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=17).split(f.iloc[outer],groups=f.iloc[outer].trip_id));train=outer[ia];valid=outer[iv]
    sets=[set(f.iloc[x].trip_id) for x in [train,valid,test]];assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    validation={};selected={};test_preds={};timings={};features=FEATURES+EXTRA
    candidates=[('Random forest',lambda:RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=42,n_jobs=-1),False)]
    if args.tabpfn:
        from tabpfn_client import TabPFNRegressor
        for variant in args.variants:
            version='v3.5-fast' if variant=='fast' else 'v3.5'
            candidates.append(('TabPFN-3.5 '+variant,lambda v=version,t=variant=='thinking':TabPFNRegressor.create_default_for_version(v,**({'thinking_mode':True} if t else {})),True))
    for name,factory,is_tab in candidates:
        model=factory();model.fit(f.iloc[train][features],f.iloc[train].remaining_min)
        options=['mean','median'] if is_tab else ['mean'];best=None
        for summary in options:
            raw=model.predict(f.iloc[valid][features],output_type=summary) if is_tab else model.predict(f.iloc[valid][features]);raw=np.maximum(0,np.asarray(raw))
            # Constant correction is fitted only on validation residuals.
            correction=float(np.median(f.iloc[valid].remaining_min.to_numpy()-raw))
            for offset in [0.,correction]:
                p=np.maximum(0,raw+offset);s=scores(f.iloc[valid].remaining_min,p,f.iloc[valid].trip_id.to_numpy());key=f'{summary}, offset {offset:.4f}'
                validation.setdefault(name,{})[key]=s
                if best is None or s['trip_balanced_mae_min']<best[0]:best=(s['trip_balanced_mae_min'],summary,offset)
        selected[name]={'summary':best[1],'offset_min':best[2],'selection_metric':'validation trip-balanced MAE'}
        model=factory();start=time.perf_counter();model.fit(f.iloc[outer][features],f.iloc[outer].remaining_min);fit_s=time.perf_counter()-start
        kwargs={'output_type':best[1]} if is_tab else {}
        start=time.perf_counter();p=np.maximum(0,np.asarray(model.predict(f.iloc[test][features],**kwargs))+best[2]);batch_s=time.perf_counter()-start
        # Same one-row cold/warm sequence for every model; network time included.
        one=f.iloc[test[:1]][features];lat=[]
        for _ in range(3):
            start=time.perf_counter();model.predict(one,**kwargs);lat.append(time.perf_counter()-start)
        test_preds[name]=p;timings[name]={'fit_seconds':fit_s,'batch_prediction_seconds':batch_s,'batch_rows':len(test),'first_single_prediction_seconds':lat[0],'warm_single_prediction_median_seconds':float(np.median(lat[1:]))}
        print(name,selected[name],scores(f.iloc[test].remaining_min,p,f.iloc[test].trip_id.to_numpy()),flush=True)
    speed=float(f.iloc[outer].recent_speed_kmh[f.iloc[outer].recent_speed_kmh.gt(3)].median());test_preds['Average-speed baseline']=np.maximum(0,f.iloc[test].distance_km.to_numpy()/speed*60)
    records=[]
    for j,(_,r) in enumerate(f.iloc[test].iterrows()):
        item=r.to_dict();item['predictions']={n:float(p[j]) for n,p in test_preds.items()};records.append(item)
    out={'status':'Validation-selected exploratory rerun; original test already inspected','dataset':'UCI GPS Trajectories, Brazil','total_bus_trips':len(set(f.trip_id)),'train_trips':len(sets[0]|sets[1]),'test_trips':len(sets[2]),'test_rows':len(test),'selection_train_trips':len(sets[0]),'validation_trips':len(sets[1]),'features':features,'selected':selected,'validation':validation,'timings':timings,'metrics':{n:scores(f.iloc[test].remaining_min,p,f.iloc[test].trip_id.to_numpy()) for n,p in test_preds.items()},'records':records,'trips':{str(int(t)):trips[t] for t in f.iloc[test].trip_id.unique()}}
    Path('dist/results-improved.json').write_text(json.dumps(out,allow_nan=False));print('Saved dist/results-improved.json. Original results.json unchanged.')
if __name__=='__main__':run()
