import numpy as np
from copy import deepcopy as dc
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

titles = ['S&P500', 'Dow Jones', 'NASDAQ']
paths = ["./stock_data//SP500_clean.csv", "./stock_data//dow_jones_clean.csv", "./stock_data//nasdaq_clean.csv"]


for i in range(len(titles)):
    data_orig = dc(np.genfromtxt(paths[i], delimiter=',', dtype=float))
    data_str = dc(np.genfromtxt(paths[i], delimiter=',', dtype=str))
    data_num = dc(data_orig[:, 1])
    data_cal = list()
    # for i in range(len(orig_data_nasdaq_str)):
    #    nasdaq_x_date.append(datetime.strptime(orig_data_nasdaq_str[i,0], '%Y-%m-%d'))
    for iv in range(len(data_str)):
        data_cal.append(datetime.strptime(data_str[iv, 0], '%Y-%m-%d'))
    # nasdaq_x_count = np.array(range(len(orig_data_nasdaq_num)))
    data_count = dc(np.array(range(len(data_num))))

    plt.plot(data_cal, data_num, color='black')
    plt.title(titles[i])
    plt.tight_layout()
    plt.show()


    # Set the style of seaborn for better visuals
    sns.set(style="whitegrid")

    # Assume that data_cal and data_num are lists or arrays with your data
    data_cal = pd.to_datetime(data_cal)  # ensure that data_cal is datetime type
    df = pd.DataFrame({"Date": data_cal, "Daily Closing Value": data_num})

    # Create the regular plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Date", y="Daily Closing Value", data=df, color="black", ax=ax)
    ax.set_title(titles[i])
    plt.tight_layout()
    plt.savefig(f'regular_plot_{titles[i]}.png', format='png')
    plt.savefig(f'regular_plot_{titles[i]}.eps', format='eps')
    plt.show()

    # Create the logarithmic plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Date", y="Daily Closing Value", data=df, color="black", ax=ax)
    ax.set_yscale("log")
    ax.set_title(titles[i])
    plt.tight_layout()
    plt.savefig(f'logarithmic_plot_{titles[i]}.png', format='png')
    plt.savefig(f'logarithmic_plot_{titles[i]}.eps', format='eps')
    plt.show()

