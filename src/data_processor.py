from src.settings import Config

import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self, hz:int|float=50, window_sec:int|float=2, overlap_pct:float=.5):
        self.hz = hz
        self.window_size = int(hz*window_sec)
        self.step_size = int(self.window_size*(1-overlap_pct))
        self.features = Config.SENSOR_FEATURES

    def clean_data(self, df):
        df[self.features] = df[self.features].interpolate(method='linear', limit=2)
        df = df.dropna(subset=self.features)
        return df

    def create_windows(self, df):
        windows = []
        labels = []
        if len(df) < self.window_size:
            return np.array([]), np.array([])
        for start in range(0, len(df) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            window_data = df[self.features].iloc[start:end].values

            label = df['Movement'].iloc[start:end].mode()[0]

            windows.append(window_data)
            labels.append(label)

        return np.array(windows), np.array(labels)

    def process_all_files(self, base_data_path):
        all_x = []
        all_y = []

        for movement_folder in os.listdir(base_data_path):
            folder_path = os.path.join(base_data_path, movement_folder)

            if os.path.isdir(folder_path):
                csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

                for file in csv_files:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    df_cleaned = self.clean_data(df)
                    x_windows, y_labels = self.create_windows(df_cleaned)

                    if x_windows.size > 0:
                        all_x.append(x_windows)
                        all_y.append(y_labels)

        return np.concatenate(all_x), np.concatenate(all_y)