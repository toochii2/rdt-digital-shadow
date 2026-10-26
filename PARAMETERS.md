# Simulator parameters

All values are illustrative. They come from published lateral-flow ranges [Parolo et al. 2020; Hsieh et al. 2017; Borse and Srivastava 2019] and were checked for plausibility against production experience. They describe no manufacturer. The source is `code/sim.py`.

## Time structure

| Item | Value |
|---|---|
| Sensor sampling | 5 min (12 samples per 1-h sub-lot), causally interpolated between hourly states |
| Batch | 8 sub-lots (one shift); 2 shifts per production day |
| Replication length | 250 production days, 4,000 sub-lots |
| Split | Chronological: training 50%, validation 20%, test 30% |
| Seeds | Design iterations used 0 to 9; all reported results use 10 to 19 |

## Nominal process

| Variable | Nominal model |
|---|---|
| Deposition ratio *f* | 1 + N(0, 0.015) per sub-lot; nozzle wear 0.15% per hour since cleaning; preventive cleaning every 32 h |
| Relative humidity (%) | 35 + 1.5 sin(diurnal) + N(0, 0.6); sensor noise SD 0.5 |
| Oven temperature (°C) | 42 + N(0, 0.5); sensor noise SD 0.3 |
| Line speed (mm/s) | Set-points 40, 50 or 60, each held for 10 to 29 h |
| Back-pressure (kPa) | 80 × (speed/50) × (1 + 2.2(1 − *f*)); sensor noise SD 1.5; glitches at a rate of 0.2% of samples (SD 25) |
| Membrane flow time (s/4 cm) | Lots of 32 to 56 h; lot value N(135, 5) clipped to 122 to 148 |
| Antibody activity | Lots of 90 to 159 h; lot value N(1.0, 0.03) clipped to 0.93 to 1.07 |
| Incoming-inspection error | Drawn once per lot: membrane SD 3.0 s/4 cm; antibody SD 0.035 |
| Missing data | 2% random sample dropout plus block outages |

## Quality outcome

| Item | Value |
|---|---|
| Intensity | *I* = *I*₀ *f* *a* *d*(*T*) *s*(RH)^−0.8 (*c*/135)^0.35; strip noise 4.5% |
| Drying factor *d* | 1 − 0.06 max(0, 40 − *T*) − 0.05 max(0, *T* − 47) |
| Spreading factor *s* | 1 + 0.035 max(0, RH − 38) |
| Line width (mm) | √*f* × *s* + N(0, 0.05) |
| Flow time (s) | 0.97 *c* + N(0, 5) (simplification: strip flow time tracks membrane flow time) |
| Specification (pass) | Intensity ≥ 0.82; width 0.75 to 1.30 mm; flow time ≤ 150 s |
| Nonconforming sub-lot | At least 2 of 5 sampled strips fail any specification; plus 1% sporadic nonconformances with no process cause |

## Injected root causes

| Cause | Duration | Severity range (severity = 1) | Expected episodes per replication |
|---|---|---|---|
| Nozzle obstruction | 12 h (4 to 10 h ramp) | *f* falls to 0.70 to 0.85 | 14 |
| Humidity excursion | 4 to 10 h (ramp, hold, recovery) | Peak 46% to 58% RH | 12 |
| Oven drift | 12 h (4 to 8 h ramp) | Down to 33 to 37 °C | 8 |
| Out-of-trend membrane lot | 16 h lot | 150 to 162 s/4 cm | 5 |
| Weak antibody lot | 16 h lot | Activity 0.80 to 0.90 | 4 |

Every training, validation and test period receives at least one episode of each type. Episodes of different types may overlap; these are concurrent-fault sub-lots.

## Evaluation constants (`code/experiments.py`)

| Constant | Value |
|---|---|
| Target false-alarm rate | 2% of validation sub-lots that passed QC |
| End-of-batch QC result | 4 h after the end of each 8-h batch |
| Response | Containment effective 1 h after the first alarm or QC result |
| Timely detection | Alarm no later than 2 h after the first nonconforming sub-lot |
| False-alarm cost | 0.5 line-hours (sensitivity analysis 0 to 4) |
| Tuning grids | LR C ∈ {0.05, 0.5, 5}; XGBoost (depth, trees) ∈ {(2, 300), (3, 300), (5, 150)}, learning rate 0.05 |

## Stress scenarios

| Scenario | Change |
|---|---|
| `sim_shared.py` | Humidity raises back-pressure; a cold oven raises the humidity reading (shared sensor channels) |
| `sim_nonlinear.py` | Humidity by drying interaction (rarely active) and a step loss of intensity when *f* < 0.88 |
| `sim_nonlinear2.py` | Humidity by line-speed interaction, active in every humidity episode and at low speed; viscous back-pressure ∝ speed^0.7, so the speed normalisation is misspecified; the same step loss |
| `experiments.py robust` | 20% sensor dropout; severity 0.7; sensor noise × 1.5 |
| `experiments.py lowprev` | Fault rates × 0.3 |
