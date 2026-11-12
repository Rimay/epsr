# EPSR DSR-Net

DSR-Net (Dynamic State-aware Routing Network) is the primary forecasting model in this branch.

## Model
- Causal TCN encoder with dilation 1, 2 and 4.
- Dynamic state descriptors: load level, first/second differences and rolling volatility.
- Next-step Normal / Peak / Ramp state classifier.
- Three specialized regression experts and soft routing using predicted state probabilities.

Future state labels are training targets and evaluation labels only. They are never passed into the forward pass.

## Ablations
- tcn: univariate load-only TCN baseline.
- state: state descriptors plus one regression head.
- state_aux: state classifier auxiliary task plus one regression head.
- full: state classifier and three routed experts.
- no_transition: routed experts without state-label cross-entropy.

## Data
UCI: data/uci/household_power_consumption.txt
GEFCom2014: data/GEFCom2014/Load/clean/gefcom2014_load_merged.csv (columns timestamp, load).

Prepare GEFCom2014 using:
python GEFCom2014_preprocess.py --input <Load-folder> --output data/GEFCom2014/Load/clean

Install:
pip install -r requirements.txt

Smoke test:
python main.py --dataset uci --freq d --models tcn state state_aux full no_transition --seeds 11 --epochs 3 --patience 2

Full experiments:
python main.py --dataset uci --freq d
python main.py --dataset GEFCom2014 --freq h --seq-len 168

Outputs in outputs/dsr/<dataset>_<freq>/ include per-seed metrics, mean/std summary, training history, prediction NPZ, best checkpoints and experiment_config.json.

Protocol: chronological 70/15/15 target-time split; scaling and Peak/Ramp thresholds fitted only on the training period. Validation/test windows can use earlier history as context. TCN baseline receives load only; other DSR variants receive all historical features. Use identical split and seeds for comparisons.

The implementation must be smoke-tested against the user's local data before claiming empirical superiority.
