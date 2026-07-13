# VisionGuard Usage Guide

## Google Colab (Recommended)
1. Open notebook in Colab
2. Enable T4 GPU
3. Run all cells

## Command Line
pip install -r requirements.txt
python inference.py --image img.jpg --category bottle

## Categories

| Category | Images | Defects |
|----------|--------|----------|
| Bottle | 209 | cracks, chips, contamination |
| Cable | 224 | bent wire, cuts, missing cable |
| Carpet | 280 | stains, cuts, holes |
| Wood | 247 | scratches, holes, stains |
| Screw | 320 | thread damage, deformation |
