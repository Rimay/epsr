# DSR-Net experiments

This experiment package is added alongside the existing RASAT-TCN/APCR-TCN code. It does not replace main.py, dataset.py, model.py, or any baseline implementation.

## 1. What is implemented

dsr_experiment.py reuses the project's existing dataset.py entry points:
- load_df(ds_name="uci" | "GEFCom2014")
- make_frame(..., freq="h" | "d", fit_frac=...)

The DSR-Net variants are:

| Variant | Architecture / purpose |
|---|---|
| tcn | Existing univariate TCN baseline |
| state | TCN + explicit dynamic-state encoder + single prediction head |
| state_aux | state + future-state auxiliary classification |
| full | state_aux + Normal/Peak/Ramp expert heads and probability-weighted fusion |
| no_transition | Full experts, but without supervised state-classification loss |
| single_head | State auxiliary task retained, expert heads replaced by one regression head |

The state vector is built from the last observed input step: standardized load level, first difference, second difference, and rolling volatility. For hourly data, volatility uses rolling_std_24; for daily data, it uses rolling_std_7.

## 2. Leakage controls

1. The existing make_frame calculates normalization statistics and Peak/Ramp thresholds on the first fit_frac portion only.
2. The future state label is attached to the target row (t+1), never included in the input window.
3. At inference, expert weights are produced only by the state classifier from historical inputs. The true future state is used only for training losses and post-hoc stratified evaluation.
4. Peak takes precedence when a target is simultaneously tagged Peak and Ramp. Valley is grouped with Normal for the three-expert design.
5. Windows are split chronologically into train/validation/test subsets.

## 3. Environment

Install the repository requirements (or a compatible environment with PyTorch, NumPy, Pandas and scikit-learn):

    pip install -r requirements.txt

Run commands from the repository root so the existing dataset.py and baseline/ imports resolve.

## 4. Run experiments

Daily UCI forecast (default; 7 historical days -> next day):

    python dsr_experiment.py --dataset uci --freq d

Hourly UCI forecast (168 hours -> next hour):

    python dsr_experiment.py --dataset uci --freq h

GEFCom2014 (expects the existing cleaned file at data/GEFCom2014/Load/clean/gefcom2014_load_merged.csv):

    python dsr_experiment.py --dataset GEFCom2014 --freq h

Override the data root if it is stored elsewhere:

    python dsr_experiment.py --dataset uci --freq d --data-root "D:/EPSR/data"

For a quick smoke run:

    python dsr_experiment.py --dataset uci --freq d --seeds 11 --epochs 3 --patience 2 --models tcn state state_aux full

For a single ablation:

    python dsr_experiment.py --dataset uci --freq d --models full --seeds 11 22 33 44 55

Optional settings include --seq-len, --batch-size, --lr, --weight-decay, --dropout, --lambda-state, and --lambda-expert.

## 5. Outputs

The default output directory is results/dsr_net/:
- metrics_by_seed.csv: Overall, Normal, Peak and Ramp test metrics for each seed/model.
- metrics_summary.csv: mean and standard deviation across seeds.
- training_history.csv: train losses and validation forecast MSE by epoch.
- pred_<dataset>_<freq>_<model>_seed<seed>.npz: standardized target/prediction and target-state labels for plots and further analysis.

MAE/RMSE are reported in the original load unit after inverse transformation. R² is computed per group; very small groups may have unstable R².

## 6. Recommended paper protocol

- First run a one-seed, three-epoch smoke test to verify paths and tensor shapes.
- Then run all variants with the same five seeds and identical chronological splits.
- Compare state vs tcn for explicit state encoding; state_aux vs state for the auxiliary state task; full vs state_aux for experts; no_transition vs full for the state-transition supervision; single_head vs full for expert specialization.
- Report overall MAE/RMSE/R² and Peak/Ramp/Normal MAE/RMSE. Also report parameter count, training time and best epoch.
- Do not claim the architecture improves generalization until the multi-seed results support it. Consider repeating the experiment on both UCI and GEFCom2014.

## 7. Important interpretation

The current dataset.py feature builder was originally designed for the existing RASAT workflow and includes calendar and reliability-related columns in its returned feature list. DSR-Net does not use the old Reliability score or RASAT gate as a model component, but the baseline feature matrix may still contain the project's engineered input columns. For a strict feature-fairness study, add a separate feature-set ablation that removes reliability-derived columns if any are added to features in a future revision.
