# EPSR: Lag-Residual TCN for Short-Term Load Forecasting

This branch replaces the previous state-classification / mixture-of-experts design. The working hypothesis is that a small chronological dataset benefits more from explicit autoregressive structure and known-future covariates than from learned state routing.

## Models

- `tcn`: unchanged univariate TCN reference.
- `lag_tcn`: multivariate TCN + explicit recent/weekly lag profile + persistence residual connection.
- `calendar_lag_tcn`: `lag_tcn` plus a separate encoder for calendar covariates known at the target timestamp (hour/day-of-week or day-of-week/month) and a compact history-shape branch.

The forecast is parameterized as the latest observed standardized load plus a learned correction. State labels are retained only for post-hoc stratified evaluation; they are not model inputs, auxiliary targets, or routing signals.

## Data protocol

Chronological 70/15/15 target-time split. Feature scaling is fitted on the training segment only. Historical input is `[t-L,t)`; target-time calendar variables are known in advance and passed separately. No target load or target-derived state is fed to the model.

## Run

```bash
pip install -r requirements.txt
python main.py --dataset uci --freq d --models tcn lag_tcn calendar_lag_tcn --seeds 11 22 33 44 55
python vis.py outputs/dsr/uci_d
```

GEFCom2014 hourly:

```bash
python main.py --dataset GEFCom2014 --freq h --seq-len 168 --models tcn lag_tcn calendar_lag_tcn --seeds 11 22 33 44 55
```

Run a short smoke test first:

```bash
python main.py --dataset uci --freq d --models tcn lag_tcn calendar_lag_tcn --seeds 11 --epochs 3 --patience 2
```

This is a testable architecture hypothesis, not a claim that it already outperforms TCN. Compare mean ± standard deviation over identical seeds and chronological splits. Report Peak/Ramp sample counts before interpreting their errors; tiny subsets are not statistically informative.
