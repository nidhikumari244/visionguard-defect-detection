# VisionGuard Results

## Benchmark Results (MVTec AD)

| Category | Image AUROC | Pixel AUROC | F1 |
|----------|:-----------:|:-----------:|:--:|
| Bottle | 0.929 | 0.954 | 0.870 |
| Cable | 0.778 | 0.915 | 0.706 |
| Carpet | 0.939 | 0.958 | 0.912 |
| Wood | 0.983 | 0.923 | 0.959 |
| Screw | 0.544 | 0.970 | 0.254 |

## Ablation Study

| Experiment | Before | After | Improvement |
|------------|--------|-------|-------------|
| Single -> Combined layers | 0.856 | 0.975 | +13.8% |
| Unnormalized -> Normalized | 0.620 | 0.778 | +25.5% |
| Threshold P95 -> P85 F1 | 0.390 | 0.706 | +81.0% |
