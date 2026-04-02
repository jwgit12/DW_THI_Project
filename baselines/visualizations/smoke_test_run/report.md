# Baseline Visualization Report

- Subjects: 1
- Methods: mppca, noisy, patch2self

## Mean Metrics Per Method

| method | mse | mae | psnr_db | runtime_s |
|---|---:|---:|---:|---:|
| mppca | 1262.245239 | 26.367598 | 30.871 | 0.111 |
| noisy | 1484.154785 | 28.704668 | 30.168 | 0.000 |
| patch2self | 4959.815918 | 44.971416 | 24.928 | 0.002 |

- Best mean PSNR method: `mppca` (30.871 dB)

## Generated Figures

- Subject-level figures are stored in `subjects/`.
- Cross-subject summary figures are stored in `summary/`.
- Full per-subject rows are stored in `metrics.csv`.
