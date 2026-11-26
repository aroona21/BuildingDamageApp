# Building Damage Assessment — Streamlit app

## Files
- `app.py`: Streamlit user interface and sliding-window inference
- `model_def.py`: ResNet50 Siamese U-Net architecture
- `requirements.txt`: Python dependencies
- `checkpoints/resnet50_damage_best.pt`: copy the trained best checkpoint here

## Run locally
1. Create a Python environment (Python 3.10 or 3.11 recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Put `resnet50_damage_best.pt` in `checkpoints/`.
4. Run: `streamlit run app.py`

The app expects a matching pre-disaster and post-disaster image. It uses 512-pixel sliding-window inference with overlap. The dashboard reports pixel-area percentages, not verified building counts.
