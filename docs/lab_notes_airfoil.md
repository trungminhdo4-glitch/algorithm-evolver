# Experiment: NASA Airfoil Self-Noise Discovery (Level 2)

## Date: 2026-02-25
## Objective: Discover physical laws (Lighthill's Law) from real NASA airfoil noise data.

### SCHRITT 0: Setup
- Created branch: `feature/nasa-airfoil-data`
- Initialized documentation.

### SCHRITT 1: Data Acquisition
- [x] Fetched data from UCI Repository.
- [x] Saved 1503 rows to `data/nasa_airfoil.csv`.

### SCHRITT 2: Universal CSV Loader
- [/] Implementation of `CsvLoaderProblem` (Added to `core/csv_problem.py`).
- [/] Integration with `DimensionalChecker`.

### SCHRITT 5: Evaluation & Findindgs
- [x] Experiment started with $10^{dB/10}$ transformation.
- [x] Multiprocessing enabled and working after pickling fixes.
- [x] Found high-magnitude constant in Pareto Front: $P \approx 7.96 \cdot 10^{12}$.
- [x] Rücktransformation in dB: $L_{dB} = 10 \cdot \log_{10}(7.96 \cdot 10^{12}) \approx 129.0 \text{ dB}$.
- [Note] The engine is finding the correct magnitude, but functional discovery ($U^5$) may require higher population or log-scaled MSE to handle the large dynamic range of the pressure ratio.

### SCHRITT 6: Commit
- [Pending] git add, commit, push.
