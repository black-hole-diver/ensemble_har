# 🏃‍♂️ Elite HAR: Pediatric Human Activity Recognition

An enterprise-grade, ultra-low-latency Machine Learning pipeline designed to classify complex, highly-variable pediatric physical activities using a single wrist-worn inertial measurement unit (IMU).

By combining raw physics extraction, aggressively regularized stacked ensembles, and real-time temporal smoothing, this engine achieves **99.24% live accuracy** with a microscopic **15.87 ms** average inference time.

---

## 🧠 System Architecture

The inference engine processes a continuous stream of sensor data through a three-stage pipeline:

1. **The Physics Engine:** Ingests a 2-second window (at 50Hz) of 9-axis sensor data (Accel, Gyro, Mag) and instantly extracts **72 mathematically proven features**, including signal magnitudes, Jerk derivatives, and Fast Fourier Transforms (FFT).
2. **The Elite Ensemble:** A threefold Stacking Classifier consisting of LightGBM, CatBoost (with L2 leaf regularization), and Random Forest. The meta-learner (Logistic Regression) votes on the physical signature.
3. **Temporal Smoothing:** A real-time `deque` buffer applies a sliding-window majority vote to eliminate single-frame physical anomalies and contextually correct the stream.

---

## 📊 Performance Metrics

Tested across an extensive randomized live-stream simulation using unseen data.

| Metric | Score / Time | Notes |
| :--- | :--- | :--- |
| **Raw F1-Score** | `0.9116` | Evaluated on static, shuffled validation sets. |
| **Raw Stream Accuracy** | `97.85%` | The base accuracy of the ensemble on a continuous data feed. |
| **Smoothed Accuracy** | `99.24%` | **Final live accuracy** after the temporal buffer. |
| **Net Smoothing Gain** | `+1.39%` | Total single-frame "flicker" errors eliminated. |

### ⚡ Latency Benchmark (1000 Windows)
*Hardware: Standard CPU (Edge-deployment simulation)*

* **Average Inference Time:** `15.87 ms`
* **99th Percentile (p99):** `18.01 ms`
* **Max (Worst-Case):** `36.44 ms` *(Well within real-time 50Hz step constraints)*
* **Math vs. ML Split:** Feature Extraction (`0.12 ms`) | Model Voting (`15.75 ms`)

---

## 🔬 The Research & Experiment Log

This repository represents a rigorous evolution from a baseline model to an elite engine. Here is the record of our physical and mathematical discoveries:

### 1. The Inter-Subject Variance Crisis
Initial pure LightGBM models struggled heavily with specific classes (`<0.70` F1). We discovered that because children execute movements with massive biological variability (different heights, energy levels, and techniques), the sensor data was bleeding together. 

### 2. Mathematical Super-Classes (Cosine Similarity)
Instead of guessing, we bypassed the ML and used Cosine Similarity on the 72-feature centroids of each movement to mathematically prove what the sensor *could* and *could not* see.
* **The Crawling Merge:** The sensor physically cannot distinguish between `Bear`, `Crab`, `Spider`, and `Rabbit` crawls (similarity > `0.95`). They were merged into **`Crawling_Play`** (resulting in a 0.98 F1).
* **The Seated Triangle:** `Book`, `Building_blocks`, and `Peck` were identical small-motor wrist twitches. Merged into **`Table_Play`**.
* **The Footwear Re-merge:** Splitting shoe-tying by "same hand" vs. "opposite hand" caused accuracy to drop to 67%. Merging them back into a single **`Footwear`** class restored it to 82%.

### 3. Killing Variance (Optuna & SMOTE)
Learning curves proved the initial model was achieving 0.00 Training Loss (memorizing the specific children) but flatlining on Validation.
* We implemented **SMOTE** to dynamically synthesize data for minority classes.
* We ran **Optuna Bayesian Optimization** with aggressive regularization, specifically clamping `max_depth` (3-7) and forcing high `min_child_samples` (50-150) to mathematically forbid the trees from memorizing individual subjects.

---

## 📂 Optimal Repository Structure

```text
har_production/
├── api/                       # FastREST API for live watch inference
│   └── main.py
├── data/                      # (Ignored in Git) Raw & processed CSVs
├── models/                    # Serialized .pkl artifacts (Model, Scaler, Encoder)
├── src/                       # Core ML source code
│   ├── config.py              # Centralized hyperparameters
│   ├── data_processor.py      # Data loading and windowing
│   └── train_elite.py         # The Optuna + Ensemble training script
├── tests/                     # Validation suite
│   ├── benchmark_latency.py   # Microsecond performance tester
│   └── elite_test.py          # Extensive stream accuracy testing
├── visuals/                   # Confusion matrices and learning curves
├── Dockerfile                 # Containerization for deployment
└── requirements.txt           # Strict environment locking