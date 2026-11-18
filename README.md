# Classifier heads in federated LoRA fine-tuning

Code and per-run results for a controlled comparison of classifier heads in federated fine-tuning of XLM-R with LoRA, on sentiment data in Marathi, English, Bengali and Hindi.

The XLM-R base encoder is frozen. Each client trains LoRA adapters (rank 8, on the query and value projections) and a classifier head, and sends both to the server every round. Six heads are compared:

| Head | Description |
|---|---|
| `linear` | mean pooling, dropout, linear layer |
| `cnn` | 1-D convolution (128 filters, width 3), max pooling |
| `bilstm` | bidirectional LSTM (256 units per direction), mean of outputs |
| `cnn_bilstm` | convolution followed by a BiLSTM |
| `mlp` | two-layer MLP with the same parameter count as `cnn_bilstm` |
| `attention` | additive attention pooling |

Federated methods: FedAvg, FedProx and FedAdam, against centralized and local-only training. Settings: even labels, Dirichlet label skew (α = 0.5 and 0.1), Marathi clients split by domain, 6, 12 and 24 clients, and partial participation. Every setting is run with three seeds.

## Main result

Test macro-F1 (mean of the four languages), 12 clients, even labels, mean ± s.d. over three seeds:

| Head | MB sent per round | FedAvg | Centralized | Local-only |
|---|---|---|---|---|
| Linear | 1.19 | 0.701 ± 0.003 | 0.708 ± 0.004 | 0.704 ± 0.002 |
| CNN | 2.36 | 0.701 ± 0.001 | 0.707 ± 0.003 | 0.704 ± 0.003 |
| BiLSTM | 9.59 | 0.701 ± 0.001 | 0.709 ± 0.001 | 0.703 ± 0.006 |
| CNN-BiLSTM | 10.77 | 0.701 ± 0.001 | 0.707 ± 0.003 | 0.704 ± 0.003 |
| MLP (matched) | 10.77 | 0.703 ± 0.002 | 0.714 ± 0.001 | 0.705 ± 0.002 |
| Attention pooling | 1.58 | 0.701 ± 0.001 | 0.712 ± 0.004 | 0.703 ± 0.001 |

The heads tie within run-to-run variation, while the heaviest sends about nine times as much per round as the linear head.

## Data

All four datasets are public, three-class (negative, neutral, positive) and used with their official splits.

| Language | Dataset | Source |
|---|---|---|
| Marathi | L3Cube MahaSent-MD | https://github.com/l3cube-pune/MarathiNLP |
| English | TweetEval (sentiment) | https://huggingface.co/datasets/cardiffnlp/tweet_eval |
| Bengali | SentNoB | https://huggingface.co/datasets/khondoker/SentNoB |
| Hindi | IIT Patna movie and product reviews (IndicGLUE) | https://huggingface.co/datasets/ai4bharat/indic_glue |

Download them into `data_raw/`; `scripts/prepare_data.py` writes the cleaned tables to `data_clean/` (normalisation, removal of conflicting duplicates and of training texts that also occur in validation or test data), and `scripts/tokenize_data.py` writes the token tables used for training.

## Layout

| Path | Contents |
|---|---|
| `src/fedheads/` | the engine: data loading and partitioning, heads, model, FedAvg, FedProx, FedAdam, local-only and centralized training, checkpointing and exact resume |
| `configs/` | default settings (`base.yaml`), selected learning rates (`tuned.yaml`), methods, client settings, and the experiment modules |
| `scripts/` | data preparation, tokenisation, running one experiment or one module, re-evaluation of saved models |
| `slurm/` | SLURM job scripts for a GPU cluster (set your account, QoS and paths) |
| `kaggle/` | single-cell Kaggle notebook runner |
| `analysis/` | scripts that build the result tables and figures from `results/` |
| `results/` | `summary.json` and per-round metrics for every run (no model weights) |
| `tests/` | unit tests |

## Running

```bash
pip install -e .
pip install -r requirements.txt

python scripts/prepare_data.py --raw data_raw --out data_clean
python scripts/tokenize_data.py --clean data_clean --out tokens

# one experiment: method in configs/method/, client setting in configs/setting/
python scripts/run_one.py --tok-dir tokens --runs-dir out/runs \
    --experiment E2 --method fedavg --setting iid12 --seed 0 --set model.head=cnn_bilstm

# one module of experiments (configs/modules/<name>.yaml)
python scripts/run_module.py --module E2_01 --tok-dir tokens --out-dir out

# tests
cd tests && python -m unittest
```

Each script prints its full options with `--help`.

To rebuild the tables and figures from the saved results:

```bash
python analysis/final_analysis.py
python analysis/make_paper_assets.py
```

## Hardware

Runs were done on NVIDIA RTX 2080 Ti GPUs (SLURM cluster) and on Kaggle T4 GPUs. A 200-round FedAvg run with 12 clients takes about one GPU hour.

## License

MIT. The datasets keep the terms of their original sources.
