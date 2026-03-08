from src.settings import SensorChannel

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set_theme(style="darkgrid", context="notebook")

def plot_movement_seaborn(file_path):
    """Plots the 9 IMU channels for a single movement file."""
    if not os.path.exists(file_path):
        print(f"Error: Could not find file at {file_path}")
        return

    df = pd.read_csv(file_path)

    movement_label = df['Movement'].iloc[0] if 'Movement' in df.columns else os.path.basename(os.path.dirname(file_path))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'IMU Sensor Data - Movement: {movement_label}', fontsize=16, fontweight='bold')

    xyz_palette = {'x': '#e74c3c', 'y': '#2ecc71', 'z': '#3498db'}

    uacc_cols = {
        SensorChannel.UACC_X: 'x',
        SensorChannel.UACC_Y: 'y',
        SensorChannel.UACC_Z: 'z'
    }
    sns.lineplot(
        data=df[list(uacc_cols.keys())].rename(columns=uacc_cols),
        palette=xyz_palette,
        ax=axes[0],
        linewidth=1.5
    )
    axes[0].set_title('User Acceleration (uacc)', loc='left')
    axes[0].set_ylabel('g')

    gyr_cols = {
        SensorChannel.GYR_X: 'x',
        SensorChannel.GYR_Y: 'y',
        SensorChannel.GYR_Z: 'z',
    }
    sns.lineplot(
        data=df[list(gyr_cols.keys())].rename(columns=gyr_cols),
        palette=xyz_palette,
        ax=axes[1],
        linewidth=1.5)
    axes[1].set_title('Gyroscope (gyr)', loc='left')
    axes[1].set_ylabel('rad/s')

    grav_cols = {
        SensorChannel.GRAV_X: 'x',
        SensorChannel.GRAV_Y: 'y',
        SensorChannel.GRAV_Z: 'z',
    }
    sns.lineplot(
        data=df[list(grav_cols.keys())].rename(columns=grav_cols),
        palette=xyz_palette,
        ax=axes[2],
        linewidth=1.5
    )
    axes[2].set_title('Gravity (grav)', loc='left')
    axes[2].set_ylabel('g')
    axes[2].set_xlabel('Time Steps')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_file = "src/movements/Ball/Ball_1.csv"
    plot_movement_seaborn(test_file)