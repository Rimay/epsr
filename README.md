# EPSR DSR-Net v2

DSR-Net is a lightweight **multi-scale, dynamic-state-aware routing network** for short-term load forecasting. The ordinary TCN baseline is retained unchanged as a reference.

## Model architecture

- **TCN baseline:** conventional causal residual TCN with dilation 1/2/4.
- **Multi-scale temporal encoder (DSR variants):** parallel causal convolution paths with dilation 1/2/4, residual channel mixing, and attention pooling over the historical window. The representation combines the latest hidden state and a history-wide context vector.
- **History-shape branch:** adaptive pooling of the historical load curve into eight bins, followed by an MLP. A learned gate fuses this compact autoregressive/history-shape representation with the multi-scale temporal representation. Adaptive pooling supports both daily and hourly sequence lengths.
- **Dynamic-state branch:** encodes historical load level, first/second differences, and rolling volatility.
- **State-aware heads:** Normal/Peak/Ramp classification and, for Full, three expert regressors softly combined by predicted state probabilities.

The forward pass consumes historical windows only. The target-time state label is used for training supervision and post-hoc evaluation, never as a model input.

## Ablations

- `tcn`: conventional univariate load-only TCN.
- `state`: DSR multi-scale encoder + history-shape branch + dynamic-state encoding + one regression head.
- `state_aux`: State plus auxiliary Normal/Peak/Ramp classification.
- `full`: State_aux plus three expert regressors and probability-based soft routing.
- `no_transition`: same routed experts, but without the state-label cross-entropy term. This is an auxiliary ablation, **not** a separate learned state-transition matrix.

## Data and runs

UCI: `data/uci/household_power_consumption.txt`

GEFCom2014: `data/GEFCom2014/Load/clean/gefcom2014_load_merged.csv` (columns `timestamp`, `load`).

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick smoke test:

```bash
python main.py --dataset uci --freq d --models tcn state state_aux full no_transition --seeds 11 --epochs 3 --patience 2
```

Full UCI daily experiment:

```bash
python main.py --dataset uci --freq d --models tcn state state_aux full no_transition --seeds 11 22 33 44 55
```

GEFCom2014 hourly experiment:

```bash
python main.py --dataset GEFCom2014 --freq h --seq-len 168
```

Results are written to `outputs/dsr/<dataset>_<freq>/`. The updated prediction figure uses one panel per model and shows the across-seed prediction standard-deviation band, avoiding the clutter of plotting all model curves on one axis.

## Experimental protocol

Chronological 70/15/15 target-time split; feature scaling and Peak/Ramp thresholds are fitted on the training segment only. Validation/test windows may use earlier observations as historical context. TCN receives load only; DSR variants receive the full historical feature set. Compare models using identical splits and random seeds.

The architecture update is a testable hypothesis, not a claim of improved accuracy. Run the local smoke test and full multi-seed experiment before drawing conclusions.
