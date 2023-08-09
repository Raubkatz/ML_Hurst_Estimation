import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from func_hurst_pretrained_model import calculate_hurst_exponent_CatBoost, calculate_hurst_exponent_CatBoost_sliding_window

window_length = 50  # You can adjust this value

# Setting the aesthetic of the plot using seaborn
sns.set_style("whitegrid")

datasets_info = [
    {
        "name": "Nile",
        "xlabel": "Year",
        "ylabel": "Volume (10^8 m^3)",
        "title": "Nile River flows at Ashwan 1871-1970"
    },
    {
        "name": "co2",
        "xlabel": "Year",
        "ylabel": "Atmospheric CO2 concentration",
        "title": "Atmospheric CO2 concentration over time"
    },
    {
        "name": "sunspots",
        "xlabel": "Year",
        "ylabel": "Number of Sunspots",
        "title": "Yearly Sunspot Activity"
    }
]

for dataset_info in datasets_info:
    data = sm.datasets.get_rdataset(dataset_info["name"]).data
    time_series_data = data['value'].values  # Convert the series to numpy array

    # Calculate the average Hurst exponent
    avg_hurst_exponent = calculate_hurst_exponent_CatBoost(time_series_data, stochastic_process='fBm')

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data['time'], time_series_data, linewidth=1.5, alpha=0.8)
    plt.title(f"{dataset_info['title']} (Hurst Exponent: {avg_hurst_exponent:.2f})", fontsize=15, fontweight='bold')
    plt.xlabel(dataset_info["xlabel"], fontsize=13)
    plt.ylabel(dataset_info["ylabel"], fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for dataset_info in datasets_info:
    data = sm.datasets.get_rdataset(dataset_info["name"]).data
    time_series_data = data['value'].values  # Convert the series to numpy array

    hurst_exponents = calculate_hurst_exponent_CatBoost_sliding_window(time_series_data, window_length, stochastic_process='fBm')

    # Offset the time to represent the last data point in the sliding window
    offset_time = data['time'].values[window_length - 1:]

    # Plotting with Twin axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    sns.lineplot(data['time'], time_series_data, ax=ax1, linewidth=1.5, alpha=0.8, color='tab:blue')
    ax1.set_xlabel(dataset_info["xlabel"], fontsize=13)
    ax1.set_ylabel(dataset_info["ylabel"], color='tab:blue', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='tab:blue', size=10)
    ax1.tick_params(axis='x', size=10)

    ax2 = ax1.twinx()
    sns.lineplot(offset_time, hurst_exponents, ax=ax2, linewidth=1.5, alpha=0.8, color='tab:red')
    ax2.set_ylabel('Hurst Exponent', color='tab:red', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='tab:red', size=10)

    plt.title(f"{dataset_info['title']} with Sliding Window Hurst Exponent", fontsize=15, fontweight='bold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

