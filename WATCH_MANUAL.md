## ⌚ Apple Watch Deployment (CoreML Export)

To support **real-time, offline activity recognition directly on Apple Watch**, we export a lightweight version of the model to Apple's **Core ML** format (`.mlpackage`). This allows the watch to perform predictions locally without requiring a phone or network connection.

Because Apple Watch devices have **strict CPU, memory, and battery constraints**, the full ensemble stack is not exported. Instead, we deploy a shallow **Random Forest** that approximates the ensemble while remaining efficient for on-device inference.

---

## 1. Export Environment Setup

Apple’s `coremltools` currently depends on **older Scikit-Learn internals**, which may cause build failures when used with the latest Python and NumPy versions. To ensure compatibility, we export the watch model using a temporary Python 3.11 environment.

Create and activate the environment:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv watch_export_env
source watch_export_env/bin/activate
```

Install compatible dependencies:

```bash
pip install scikit-learn==1.2.2 coremltools "numpy<2" pandas
```

Run the export script:

```bash
python -m src.export_watch_model
```

Once the export is complete, deactivate and remove the temporary environment:

```bash
deactivate
rm -rf watch_export_env
```

---

## 2. Model Optimization

During conversion, `coremltools` automatically performs **Float16 precision reduction**, reducing the memory footprint by roughly **50%** while preserving prediction accuracy. This optimization is especially useful for resource-constrained devices such as Apple Watch.

Additionally, the exported model is trained using **human-readable string labels** (e.g., `CRAWLING_PLAY`, `RUNNING`, `JUMPING`). These labels are embedded directly into the `.mlpackage`, allowing the watchOS app to receive readable predictions without additional label-mapping logic in Swift.

---

## 3. Integration in Xcode

After export, the file:

```
models/WatchActivityPredictor.mlpackage
```

can be dragged directly into the **Watch App Extension target** in Xcode. Xcode will automatically compile the model and generate a corresponding Swift interface.

---

## 4. watchOS Inference Pipeline

The watchOS application performs the following steps:

1. Collect **accelerometer and gyroscope data** using `CoreMotion`
2. Buffer **2 seconds of sensor data at 50Hz**
3. Compute the **72 engineered physics features**
   - Means
   - Standard deviations
   - Frequency-domain (FFT) features
4. Pass the feature vector to the model as:

```swift
MLMultiArray(shape: [1, 72], dataType: .double)
```

5. Receive the predicted activity label directly from Core ML.

This pipeline enables **low-latency activity recognition running entirely on-device**.