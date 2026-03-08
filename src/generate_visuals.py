from src.settings import SensorChannel, Config

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid", context="notebook")
xyz_palette = {'x': '#e74c3c', 'y': '#2ecc71', 'z': '#3498db'}

def save_movement_plot(csv_path, save_path, movement_label):
    """Generates and saves the 9-channel IMU plot for a given CSV."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'IMU Sensor Data - Movement: {movement_label}', fontsize=16, fontweight='bold')

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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def process_all_visuals(input_base_dir=Config.RAW_DATA_DIR, output_base_dir=f"{Config.VISUALS_DIR}/movement_visualizations"):
    """Iterates through all movement folders and generates plots."""
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    categories = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]

    print(f"Found {len(categories)} movement categories. Starting batch generation...")

    for category in categories:
        cat_input_dir = os.path.join(input_base_dir, category)
        cat_output_dir = os.path.join(output_base_dir, category)

        if not os.path.exists(cat_output_dir):
            os.makedirs(cat_output_dir)

        csv_files = [f for f in os.listdir(cat_input_dir) if f.endswith('.csv')]

        print(f"Processing '{category}' ({len(csv_files)} files)...")

        for file in csv_files:
            input_file_path = os.path.join(cat_input_dir, file)

            output_file_name = file.replace('.csv', '.png')
            output_file_path = os.path.join(cat_output_dir, output_file_name)

            save_movement_plot(input_file_path, output_file_path, movement_label=category)

    print(f"\nSuccess! All visuals have been saved to the '{output_base_dir}' directory.")

if __name__ == "__main__":
    process_all_visuals()