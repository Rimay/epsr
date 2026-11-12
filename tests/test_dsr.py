import numpy as np
import pandas as pd
from dataset import make_frame, WindowDataset, chronological_split
from model import DSRNet

def test_all_variants_forward():
    n=1200
    idx=pd.date_range("2020-01-01",periods=n,freq="h")
    t=np.arange(n)
    load=2+0.3*np.sin(2*np.pi*t/24)+0.05*np.random.default_rng(7).normal(size=n)
    frame,features,meta=make_frame(pd.DataFrame({"load":load},index=idx),"h")
    ds=WindowDataset(frame,features,48)
    parts=chronological_split(ds,meta["train_end"],meta["val_end"])
    assert all(len(p)>0 for p in parts)
    x,_,_=ds[0]
    for variant in DSRNet.VARIANTS:
        dim=1 if variant=="tcn" else len(features)
        indices=[0] if variant=="tcn" else [features.index(c) for c in
            ["load","delta","delta2","rolling_std_24"]]
        model=DSRNet(dim,indices,variant=variant)
        xx=x[None,:,:1] if variant=="tcn" else x[None,:,:]
        out=model(xx)
        assert out["pred"].shape==(1,1)
        if variant in {"full","no_transition"}:
            assert out["expert_preds"].shape==(1,3)
            assert out["state_probs"].shape==(1,3)
        if variant in {"state_aux","full","no_transition"}:
            assert out["state_logits"].shape==(1,3)
