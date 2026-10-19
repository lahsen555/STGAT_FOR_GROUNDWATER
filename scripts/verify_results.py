"""Verify saved research results without retraining; optionally replay checkpoints."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'code/model_runs'
KEY=['origin_month','target_month','lead_months','well_id','is_observed','observed_m']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints',action='store_true',help='Reload all 18 neural and 3 no-GAT checkpoints and compare forecasts; no fitting.')
    args=parser.parse_args()
    manifest=ROOT/'docs/file_manifest.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text())['sha256'].items():
            if digest(ROOT/name)!=expected:raise ValueError(f'Packaged file changed: {name}')
    for run in ['optimized_h1_20261015','optimized_20261015']:
        path=RUNS/run;record=json.loads((path/'run.json').read_text())
        assert record['complete'] and record['selection_frozen']
        for name,expected in record['input_sha256'].items():assert digest(ROOT/name)==expected,name
        for name,expected in record['source_sha256'].items():assert digest(path/'source_snapshot'/name)==expected,name
    report=RUNS/'optimized_report_20261015'
    frame=pd.read_csv(report/'recommended_predictions.csv')
    assert len(frame)==540 and int(frame.is_observed.sum())==389
    assert frame.loc[frame.well_id=='1580/52','is_observed'].sum()==0
    raw=pd.read_csv(ROOT/'data/gwl_and_features_advanced_sumed.csv')
    truth=raw[['well_id','date','level']].rename(columns={'date':'target_month','level':'raw_m'})
    joined=frame.merge(truth,on=['well_id','target_month'],validate='one_to_one')
    np.testing.assert_allclose(joined.observed_m,joined.raw_m,equal_nan=True)
    observed=frame.loc[frame.is_observed]
    expected=pd.read_csv(report/'recommended_metrics.csv').set_index('model')
    for model in ['Persistence','Seasonal','DampedTrend','STGAT','GRU','LSTM']:
        error=observed[model]-observed.observed_m
        mae=abs(error).mean();rmse=np.sqrt(np.mean(error**2));r2=[]
        for _,g in observed.groupby('well_id'):
            denom=((g.observed_m-g.observed_m.mean())**2).sum()
            if len(g)>=2 and denom>1e-12:r2.append(1-((g[model]-g.observed_m)**2).sum()/denom)
        np.testing.assert_allclose([mae,rmse,np.mean(r2)],expected.loc[model,['mae_m','rmse_m','macro_r2']].to_numpy(dtype=float),atol=1e-10,rtol=0)
    print('PASS: file/input/source hashes and all saved one-month metrics; 389 measured targets, 19 scored wells.')
    if not args.checkpoints:return
    sys.path.insert(0,str(ROOT/'code'))
    import torch
    import optimize_stgat as opt
    from stgnn_data import spatial_graph
    torch.set_num_threads(2)
    for run in ['optimized_h1_20261015','optimized_20261015']:
        folder=RUNS/run;record=json.loads((folder/'run.json').read_text())
        factory=opt.DataFactory(raw,pd.read_csv(ROOT/'data/wells_meta.csv'),horizon=record['forecast_horizon'])
        stored=pd.read_csv(folder/'all_forecasts.csv')
        for kind in ['STGAT','GRU','LSTM']:
            cfg=opt.OptimConfig(**record['models'][kind]['config'])
            data=factory.get(cfg,record['training_end'],record['selection_end'])
            starts=np.arange(24,len(data.dates));members=[]
            for seed in record['ensemble_seeds']:
                cp=torch.load(folder/f'{kind}_seed{seed}.pt',map_location='cpu',weights_only=True)
                assert cp['preprocessing']==data.preprocessing(),'Checkpoint preprocessing differs'
                model=opt.CompactForecast(kind,data,cfg);model.load_state_dict(cp['state_dict'])
                members.append(opt.forecast(model,data,starts,cp['edge_index'],cp['edge_attr'],'cpu'))
            replay=opt.rows_for_predictions(data,starts,{kind:np.mean(members,axis=0)})
            pd.testing.assert_frame_equal(replay[KEY],stored[KEY])
            np.testing.assert_allclose(replay[kind],stored[kind],atol=1e-5,rtol=0)
            print(f'PASS: {run} {kind}, all 3 seeds reloaded, maximum difference {abs(replay[kind]-stored[kind]).max():.3g} m')
    folder=ROOT/'article/analysis/no_gat';record=json.loads((folder/'protocol.json').read_text())
    cfg=opt.OptimConfig(**record['config'])
    data=opt.DataFactory(raw,pd.read_csv(ROOT/'data/wells_meta.csv'),horizon=1).get(cfg,'2015-08-01','2017-02-01')
    edge,attr,_=spatial_graph(data.coordinates,cfg.k)
    edge=torch.as_tensor(edge);attr=torch.as_tensor(attr)
    starts=np.arange(24,len(data.dates));members=[]
    for seed in record['seeds']:
        cp=torch.load(folder/f'seed{seed}.pt',weights_only=True,map_location='cpu')
        assert cp['preprocessing']==data.preprocessing()
        model=opt.CompactForecast('GRU',data,cfg);model.load_state_dict(cp['state_dict'])
        members.append(opt.forecast(model,data,starts,edge,attr,'cpu'))
    replay=opt.rows_for_predictions(data,starts,{'NoGAT':np.mean(members,axis=0)})
    stored=pd.read_csv(folder/'all_forecasts.csv')
    pd.testing.assert_frame_equal(replay[KEY],stored[KEY])
    np.testing.assert_allclose(replay.NoGAT,stored.NoGAT,atol=1e-5,rtol=0)
    print('PASS: all 21 checkpoints reproduce their saved ensemble forecasts; no training or benchmark selection.')

if __name__=='__main__':main()
