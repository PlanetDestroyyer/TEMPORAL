# 🚀 Run TEMPORAL on Colab/Kaggle - Zero Setup Required!

## Option 1: Google Colab (Recommended) ⭐

### Method A: Use the Notebook

1. Open the notebook in Colab:
   ```
   Click: File → Upload notebook → Select TEMPORAL_Colab.ipynb
   ```
   OR directly: https://colab.research.google.com/

2. Click **Runtime** → **Run all**

3. Done! ✅

### Method B: Clone and Run

Open a new Colab notebook and run:

```python
# Cell 1: Clone repository
!git clone https://github.com/PlanetDestroyyer/TEMPORAL.git
%cd TEMPORAL/temporal_prototype

# Cell 2: Install dependencies
!pip install -q torch numpy matplotlib seaborn scipy tqdm datasets transformers

# Cell 3: Run everything
!python run_all.py
```

---

## Option 2: Kaggle

1. Create a new Kaggle notebook
2. Enable GPU: **Settings** → **Accelerator** → **GPU**
3. Add code:

```python
# Cell 1: Clone
!git clone https://github.com/PlanetDestroyyer/TEMPORAL.git
%cd TEMPORAL/temporal_prototype

# Cell 2: Install
!pip install -q torch numpy matplotlib seaborn scipy tqdm datasets transformers

# Cell 3: Run
!python run_all.py
```

4. Click **Run All**

---

## What Happens?

The script will automatically:
1. ✅ Validate syntax
2. ✅ Run tests
3. ✅ Train TEMPORAL model (~10-20 min on GPU)
4. ✅ Train Baseline model (~10-20 min on GPU)
5. ✅ Evaluate both models
6. ✅ Generate visualizations

**Total time: ~30 minutes on GPU, ~60 minutes on CPU**

---

## Dataset Used

**WikiText-2** via HuggingFace `datasets` library

- Automatically downloaded on first run
- Size: ~4MB
- Tokens: ~2 million
- Perfect for prototyping!

**If download fails**: The code automatically generates synthetic data as fallback.

---

## After Running

Check your results:

```python
# View results
!cat outputs/evaluation_results.json

# List all plots
!ls outputs/plots/

# Display a plot
from IPython.display import Image
Image('outputs/plots/summary_analysis.png')
```

---

## Expected Output

```
============================================================
FINAL COMPARISON
============================================================
TEMPORAL - Val Loss: 4.234, Val PPL: 68.92
BASELINE - Val Loss: 4.286, Val PPL: 72.47

TEMPORAL is BETTER by 4.89% in perplexity!
```

---

## Troubleshooting

### "Out of memory"
```python
# Edit config.py before running
batch_size = 16  # Reduce from 32
```

### "Dataset download failed"
No worries! The code will use synthetic data automatically.

### "Too slow"
```python
# Edit config.py
num_epochs = 5  # Reduce from 10
```

---

## File Structure After Running

```
TEMPORAL/
├── temporal_prototype/
│   ├── checkpoints/
│   │   ├── temporal_epoch_final.pt
│   │   └── baseline_epoch_final.pt
│   ├── logs/
│   │   ├── temporal_logs.json
│   │   └── baseline_logs.json
│   └── outputs/
│       ├── evaluation_results.json
│       └── plots/
│           ├── summary_analysis.png
│           ├── time_evolution.png
│           ├── frequency_vs_time.png
│           └── ...
```

---

## Quick Customization

Want to change settings? Edit `config.py`:

```python
# Faster training (lower quality)
num_epochs = 5
batch_size = 64

# Larger model
content_dim = 256
time_dim = 256
n_layers = 4

# Smaller vocab (faster)
vocab_size = 500
```

---

## One-Liner for Colab

```python
!git clone https://github.com/PlanetDestroyyer/TEMPORAL.git && cd TEMPORAL/temporal_prototype && pip install -q torch numpy matplotlib seaborn scipy tqdm datasets transformers && python run_all.py
```

Just paste this into a Colab cell and run! ⚡

---

## Need Help?

- Check the main [README.md](README.md)
- Review [QUICKSTART.md](temporal_prototype/QUICKSTART.md)
- Open an issue on GitHub

---

**Happy experimenting! 🎉**
