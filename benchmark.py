"""Trip-disjoint remaining-time benchmark. Run from the project directory."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

FEATURES=['latitude','longitude','destination_lat','destination_lon','distance_km','elapsed_min','recent_speed_kmh','hour','weekday']
def haversine(lat,lon,lat2,lon2):
    a=np.sin(np.radians(lat2-lat)/2)**2+np.cos(np.radians(lat))*np.cos(np.radians(lat2))*np.sin(np.radians(lon2-lon)/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
def prepare():
    meta=pd.read_csv('data/go_track_tracks.csv');pts=pd.read_csv('data/go_track_trackspoints.csv')
    ids=set(meta.loc[meta.car_or_bus.eq(2),'id']);rows=[];trips={}
    for tid,g in pts[pts.track_id.isin(ids)].groupby('track_id'):
        g=g.copy();g['time']=pd.to_datetime(g.time);g=g.sort_values('time').drop_duplicates('time').reset_index(drop=True)
        if len(g)<15:continue
        duration=(g.time.iloc[-1]-g.time.iloc[0]).total_seconds()/60
        if duration<3 or duration>180:continue
        dest=g.iloc[-1];elapsed=(g.time-g.time.iloc[0]).dt.total_seconds()/60
        segment=haversine(g.latitude.shift(),g.longitude.shift(),g.latitude,g.longitude).fillna(0)
        delta=g.time.diff().dt.total_seconds()/3600
        speed=(segment/delta).replace([np.inf,-np.inf],np.nan).clip(0,120).rolling(6,min_periods=1).mean().fillna(0)
        dist=haversine(g.latitude,g.longitude,dest.latitude,dest.longitude)
        sampled=list(range(6,len(g)-1,max(1,len(g)//40)))
        for i in sampled:
            if duration-elapsed.iloc[i]<0.5:continue
            r={'trip_id':int(tid),'latitude':float(g.latitude.iloc[i]),'longitude':float(g.longitude.iloc[i]),'destination_lat':float(dest.latitude),'destination_lon':float(dest.longitude),'distance_km':float(dist.iloc[i]),'elapsed_min':float(elapsed.iloc[i]),'recent_speed_kmh':float(speed.iloc[i]),'hour':int(g.time.iloc[i].hour),'weekday':int(g.time.iloc[i].weekday()),'remaining_min':float(duration-elapsed.iloc[i])}
            rows.append(r)
        trips[int(tid)]={'duration_min':duration,'path':g[['latitude','longitude']].iloc[::max(1,len(g)//180)].values.tolist()}
    return pd.DataFrame(rows),trips

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tabpfn',action='store_true');args=parser.parse_args()
    frame,trips=prepare();train,test=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42).split(frame,groups=frame.trip_id))
    assert not set(frame.iloc[train].trip_id)&set(frame.iloc[test].trip_id)
    X=frame[FEATURES];y=frame.remaining_min
    median_speed=float(frame.iloc[train].recent_speed_kmh[frame.iloc[train].recent_speed_kmh.gt(3)].median())
    baseline=np.maximum(0,frame.iloc[test].distance_km.to_numpy()/median_speed*60)
    rf=RandomForestRegressor(n_estimators=200,min_samples_leaf=5,random_state=42,n_jobs=-1)
    start=time.perf_counter();rf.fit(X.iloc[train],y.iloc[train]);pred=np.maximum(0,rf.predict(X.iloc[test]));rf_time=time.perf_counter()-start
    predictions={'Average-speed baseline':baseline,'Random forest':pred};timings={'Random forest':rf_time}
    if args.tabpfn:
        from tabpfn_client import TabPFNRegressor
        model=TabPFNRegressor.create_default_for_version('v3.5')
        start=time.perf_counter();model.fit(X.iloc[train],y.iloc[train]);predictions['TabPFN-3.5 Plus']=np.maximum(0,model.predict(X.iloc[test]));timings['TabPFN-3.5 Plus']=time.perf_counter()-start
    metrics={}
    for name,p in predictions.items():
        errors=np.abs(p-y.iloc[test].to_numpy());trip_errors=pd.DataFrame({'trip':frame.iloc[test].trip_id.values,'err':errors}).groupby('trip').err.mean()
        metrics[name]={'mae_min':float(errors.mean()),'trip_balanced_mae_min':float(trip_errors.mean()),'within_2_min':float((errors<=2).mean()),'fit_predict_seconds':timings.get(name)}
    records=[]
    for j,(_,r) in enumerate(frame.iloc[test].iterrows()):
        item=r.to_dict();item['predictions']={name:float(p[j]) for name,p in predictions.items()};records.append(item)
    output={'status':'TabPFN evaluated' if args.tabpfn else 'Baseline results only; TabPFN not run','dataset':'UCI GPS Trajectories, Brazil','total_bus_trips':int(frame.trip_id.nunique()),'train_trips':int(frame.iloc[train].trip_id.nunique()),'test_trips':int(frame.iloc[test].trip_id.nunique()),'train_rows':len(train),'test_rows':len(test),'metrics':metrics,'records':records,'trips':{str(t):trips[t] for t in frame.iloc[test].trip_id.unique()},'median_training_speed_kmh':median_speed}
    Path('dist').mkdir(exist_ok=True);Path('dist/results.json').write_text(json.dumps(output,allow_nan=False));print(json.dumps({k:v for k,v in output.items() if k not in ['records','trips']},indent=2))
if __name__=='__main__':main()
