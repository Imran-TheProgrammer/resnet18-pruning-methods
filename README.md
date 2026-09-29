#
The notebook implements **Tasks 0–3** and a **Final Report** section: dense baseline profiling, unstructured magnitude pruning, iterative GraSP pruning, structured channel pruning, result tables, plots, and short discussions. 

Run cells **top to bottom** in order: shared setup → Task 0 → Task 1 → Task 2 → Task 3 → Final Report.

---

## Requirements

- **Python 3.11** (3.10+ may work)
- **CUDA GPU** recommended (the notebook uses GPU training and profiling)
- **Intel RAPL** (CPU energy) and **NVML** (GPU power) for profiling cells from Task 0.3 onward—use the same machine for all tasks

Python packages are listed in [`requirements.txt`](requirements.txt). **PyTorch** is not pinned there because the wheel must match your CUDA driver.

---

## Environment setup

From the project directory:

```bash
cd /path/to/[Folder]
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch (adjust the index URL for your CUDA version):

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Register a Jupyter kernel for this venv:

```bash
python -m ipykernel install --user --name resnet-pruning --display-name "ResNet Pruning"
```

---

## Run the assignment

1. Start Jupyter from the project folder with the venv activated:

   ```bash
   source .venv/bin/activate
   jupyter notebook main.ipynb
   ```

2. In the notebook, choose kernel **ResNet Pruning**.

3. Use **one kernel session** for the whole run. After *Kernel → Restart*, run the **shared setup** cells again before continuing.

4. Execute in this order (section headers match the notebook):

   | Order | Notebook sections |
   |------:|---------------------|
   | 1 | Bootstrap + **Shared setup** (imports, model, data, training, profiling helpers) |
   | 2 | **Task 0** — load checkpoint, evaluate, profile dense model |
   | 3 | **Task 1** — sensitivity, local/global pruning, fine-tuning, COO / sparse inference checks |
   | 4 | **Task 2** — calibration, warm-up, iterative magnitude and GraSP, final comparison |
   | 5 | **Task 3** — structured channel pruning, fine-tuning, profile |
   | 6 | **Final Report** — run the helper cell (`load_required_json`, `run_report`), then the report execution cell |

5. Long steps (especially Task 1.2 sensitivity and Task 2 training) can take hours on first run. The notebook saves progress automatically; re-run the same cell after an interrupt to continue (default resume is on).

6. **Save the notebook** with all outputs visible



---

## Profiling on Linux (Task 0.3+)

If profiling fails with a **PermissionError** on Intel RAPL energy files, allow read access once (then restart the kernel and re-run setup + Task 0.3):

```bash
sudo chmod a+r /sys/devices/virtual/powercap/intel-rapl/intel-rapl:*/energy_uj
sudo chmod -R a+r /sys/devices/virtual/powercap/intel-rapl/intel-rapl:*/intel-rapl:*/energy_uj
```

---
