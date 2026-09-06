# Reproduction: Yield-Aware Trajectory Optimization for Wafer Stages in Step-and-Scan Lithography Systems (CNCA)

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested with mujoco 3.11, torch 2.13. Device selection: cuda, then mps, then cpu.

WM-811K labeled subset (Wu et al., 2015): Kaggle dataset
`qingyi/wm811k-wafer-map`, file `LSWMD.pkl` (~2.1 GB). Pass its path with
`--data`.

## Experiments

Output directory defaults to `figures/` (`--out`).

### CNN training (Tables dataset, hw, cnnarch)

```
python3 run.py train-cnn --data /path/to/LSWMD.pkl
```

Per-class sample counts (Table dataset) and test accuracy after each of 5
epochs (95.68% / 96.56% / 96.61% / 96.85% / 96.90%). Saves
`common/wafer_cnn.pth`, required by `pso e4`.

### Controller ladder (Table ladder, Figure controllers_esyn)

```
python3 run.py controllers --out figures
```

Case 1 (long-stroke PD) -> Case 2 (+ lag feedforward) -> Case 3a (+
short-stroke PI) -> Case 3b (+ short-stroke HIGS), baseline trajectory
(v=0.8 m/s, a=20 m/s^2, j=1600 m/s^3, s=1e5 m/s^4), 4-die row.

### Profile order (Table order)

```
python3 run.py order --out figures
```

2nd/3rd/4th order and 4th-order tight-snap (s=2e4) profiles, under Case 1
and Case 3b.

### PSO scenarios (Table pso, Figure pso_map)

```
python3 run.py pso e3 --config case3b --seed 0 --out figures
python3 run.py pso e4 --config case3b --seed 0 --out figures
python3 run.py pso e4 --config case3b --seed 1 --out figures
python3 run.py pso e4 --config case3b --seed 0 --init log --gamma 0.5 --out figures
```

E3, E4 (seed 0), E4 (seed 1), E4'. Baseline row is the first evaluated
particle of any run (v=0.8, a=20, j=1600, s=1e5). `e4`/`e4'` require
`common/wafer_cnn.pth`. 48 evaluations per run (6 particles x 8
iterations); checkpoints after every iteration in `figures/.pso_ckpt_*.pkl`.

## Layout

```
common/
  profiles.py     2nd/3rd/4th-order bounded-derivative motion profiles
  litho_model.py  16-body MuJoCo scanner model
  controllers.py  Case 1/2/3a/3b feedback loops
  cnn_model.py    CNN architecture
  cnn_cost.py     wafer-map classification and yield cost
  headless.py     mujoco.viewer stub for headless execution
experiments/
  train_cnn.py
  controller_ladder.py
  profile_order.py
  pso_scenarios.py
run.py
```
