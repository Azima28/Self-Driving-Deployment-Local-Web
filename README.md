# FCN8s Segmentation Web App

This repository contains a Flask web interface for running semantic segmentation using FCN8s / U-Net models implemented in PyTorch.

**Important:** Model weight files (`.pth`) are large and kept in the `model/` folder locally; they are excluded from version control via `.gitignore`.

## Structure
- `main/` - Flask app and templates
  - `app.py` - main Flask application (do not modify model code unless you know what you're doing)
  - `templates/` - HTML templates
  - `uploads/`, `outputs/` - runtime directories (ignored)
- `model/` - model weights (kept locally, not checked into git)

## Quick start (Windows, Anaconda)
1. Create or activate environment (you already have `pytorch`):

```powershell
conda activate pytorch
```

2. Install dependencies (if not already installed):

```powershell
pip install -r main/requirements.txt
```

3. Run the app:

```powershell
cd main
python app.py
```

4. Open http://127.0.0.1:5000 in your browser.

## GPU notes
- To use GPU with PyTorch, you must have an NVIDIA GPU and install PyTorch with CUDA support. Check CUDA availability:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

- If `False`, reinstall PyTorch with the proper CUDA build matching your drivers. Example for CUDA 11.8:

```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```
