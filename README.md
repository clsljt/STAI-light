# STAI

Official PyTorch implementation of **STAI** for unsupervised multivariate time-series anomaly detection.

STAI is a compact masked-reconstruction model (about 0.18 M parameters with the default configuration). Each channel
is embedded independently in the time domain, while a small set of learnable global tokens summarises the spectrum of
every channel's window and exchanges information across channels. Every time step then reads this spectral memory
through a gated cross-attention before the window is reconstructed.

```
spectral path   x -> InstanceNorm -> rFFT -> bin tokens [Re, Im, log(1+|X|)]
                  -> K learnable global tokens cross-attend to the bins            => memory (per channel, K x D)
                  -> channel fusion: for each global token, attention across channels
temporal path   x -> RevIN -> per-channel convolutional embedding                  => tokens (per channel, L x D)
fusion          each time step cross-attends to the memory of its channel (gated residual + FFN)
decoder         linear map over time (L -> L), linear D -> 1, RevIN de-normalisation
```

One configuration (`configs/stai_lite.json`) is used for every dataset; there is no per-dataset tuning.

## Requirements

```
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.4.1 (CUDA 12.1), NumPy 2.2, pandas 2.3, scikit-learn 1.7 on one NVIDIA A100.

## Data

We use the pre-processed multivariate anomaly-detection datasets of the CATCH / TAB benchmark
([CATCH](https://github.com/decisionintelligence/CATCH), [TAB](https://github.com/decisionintelligence/TAB)), with
their train / test boundaries (`train_lens` in `DETECT_META.csv`). The pre-processed data can be downloaded from

* CATCH: [OneDrive](https://1drv.ms/u/c/801ce36c4ff3f93b/EVTDLHyvegpEn_Oxa6ZiuFIBjTsKk6m9JldUqWDqvrVCnQ?e=P2T3Vc) or
  [BaiduCloud](https://pan.baidu.com/s/1W7UoAWKZjoukSZ74FTipYA?pwd=2255)
* TAB: [Google Drive](https://drive.google.com/file/d/1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT/view?usp=drive_link)

Take the CSV files of the eight datasets from the anomaly-detection folder of the archive and place them as follows:

```
dataset/
├── DETECT_META.csv          # provided here (rows of the eight datasets used in the paper)
└── data/
    ├── MSL.csv  PSM.csv  SMAP.csv  SMD.csv  SWAT.csv
    └── GECCO.csv  Genesis.csv  Creditcard.csv
```

Each CSV is read once and cached as `.npy` under `dataset/cache/`.

## Usage

Train and test one dataset with one seed:

```
python run.py --data MSL --seed 2026 --gpu 0
```

All datasets with seeds 2026, 2027 and 2028, followed by the mean and standard deviation over seeds:

```
bash scripts/run_all.sh 0
```

Outputs are written to `results/<dataset>/seed_<seed>/`: `checkpoint.pth` (selected epoch), `train_log.json`,
`test_scores.npy` and `metrics.json`. `python scripts/summarize.py` prints the table over seeds.

## Protocol

* **Split.** The training part is `[0, train_lens)` of each series. Its last 20 % is held out for validation; the
  scaler (per-channel mean and standard deviation) is fitted on the first 80 % only.
* **Training.** Windows of 192 steps with stride 4. In every window 6 random patches of 8 steps are masked (set to 0)
  in all channels, and the loss is the masked MSE + 0.1 x the MSE over the window + 0.05 x the mean squared error of
  the (unnormalised) rFFT. AdamW, learning rate 1e-3 with a one-cycle schedule over 20 epochs, mixed precision,
  gradient clipping at 1.0.
* **Model selection.** Early stopping with patience 5 on the validation loss; the epoch with the lowest validation
  loss is tested. Test labels are not used for any choice.
* **Scoring.** The test part is cut into non-overlapping 192-step windows (an incomplete tail is not scored). Each
  window is reconstructed four times with complementary masks, so that every time step is masked exactly once; its
  score is the squared reconstruction error averaged over channels plus 0.05 x the spectral error of that window.
* **Metrics.** AUC-ROC, AUC-PR, R-AUC-ROC, R-AUC-PR, VUS-ROC and VUS-PR, computed with the code of the CATCH / TAB
  benchmark (`metrics/vus_metrics.py`, copied unchanged); the sliding window of the range-based measures is the median
  length of the anomaly segments, and twice that for VUS.

## Repository layout

```
run.py                          entry point
configs/stai_lite.json          the single configuration used for all datasets
data_provider/data_loader.py    reading the benchmark CSVs, split, scaler, windows
models/STAI.py                  the model
exp/exp_anomaly_detection.py    training, model selection, scoring, evaluation
metrics/                        the six measures (VUS code from CATCH / TAB)
scripts/                        run_all.sh, summarize.py
```

## Acknowledgements

The datasets, splits and evaluation code follow CATCH (Wu et al., ICLR 2025) and TAB (Qiu et al., PVLDB 2025). The
range-based and VUS measures are those of Paparrizos et al. (PVLDB 2022).

## Citation

```
@article{stai,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```
