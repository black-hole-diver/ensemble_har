from src.settings import Config, FileNames
from src.utils import extract_physics, clean_class_name

import os
import numpy as np
import joblib
from collections import deque, Counter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Elite HAR Inference API",
    description="Real-time Human Activity Recognition using stacked ensembles and temporal smoothing.",
    version="1.0.0"
)

class SensorWindow(BaseModel):
    data: list[list[float]] = Field(
        ...,
        description="A 2D array representing one 2-second window. Axes: Accel(x,y,z), Gyro(x,y,z), Mag(x,y,z)."
    )

class LiveSmoothedPredictor:
    def __init__(self, buffer_size=3):
        print("--- Booting Elite Inference Engine ---")
        try:
            model_path = os.path.join(Config.MODELS_DIR, FileNames.MODEL_NAME)
            scaler_path = os.path.join(Config.MODELS_DIR, FileNames.SCALER_NAME)
            labels_path = os.path.join(Config.MODELS_DIR, FileNames.LABELS_NAME)

            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(labels_path)
        except FileNotFoundError as e:
            raise RuntimeError(f"❌ Critical ML artifact missing: {e}. Check your MODELS_DIR.")

        self.prediction_buffer = deque(maxlen=buffer_size)

    def predict(self, window_data: np.ndarray):
        features = extract_physics(window_data).reshape(1, -1)

        scaled = self.scaler.transform(features)
        pred_encoded = self.model.predict(scaled)
        raw_pred = self.label_encoder.inverse_transform(pred_encoded)[0]

        clean_raw = clean_class_name(raw_pred)

        self.prediction_buffer.append(clean_raw)
        if len(self.prediction_buffer) == self.prediction_buffer.maxlen:
            smooth_pred = Counter(self.prediction_buffer).most_common(1)[0][0]
        else:
            smooth_pred = clean_raw
        return clean_raw, smooth_pred

    def reset_buffer(self):
        self.prediction_buffer.clear()

engine = LiveSmoothedPredictor()

@app.post("/predict")
async def process_window(payload: SensorWindow):
    try:
        window_array = np.array(payload.data)

        if window_array.shape[1] != 9:
            raise HTTPException(status_code=400, detail=f"Expected 9 sensor axes, got {window_array.shape[1]}")

        raw_pred, smooth_pred = engine.predict(window_array)

        return {
            "status": "success",
            "raw_activity": raw_pred,
            "smoothed_activity": smooth_pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
async def reset_stream():
    """Call this when a new recording session starts on the watch to clear memory."""
    engine.reset_buffer()
    return {"status": "success", "message": "Temporal buffer cleared."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)