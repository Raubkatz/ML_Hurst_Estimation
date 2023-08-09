
import hurst
from copy import deepcopy as dc
import flm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
from matplotlib import dates as mdates




alphas = [0.5, 1.0, 1.5, 2.0]

scaling_exponent = 0.5


#max_len = 12317
max_len = 178
#step_size = 1000
step_size = 150
plot_assets = False

av_log_data_finance = np.zeros((3+1,100))
av_log_data_finance[1:,0]  = [0,1,2]
std_log_data_finance = np.zeros((3+1,100))
std_log_data_finance[1:,0] = [0,1,2]

titles = ['S&P500', 'Dow Jones', 'NASDAQ']
x_SP = list()
x_DJ = list()
x_NA = list()
y_SP = list()
y_DJ = list()
y_NA = list()


hurst_av = list()
hurst_std = list()
for i in range(3):
    finance_data = i
    if finance_data == 0:
        data_orig = dc(np.genfromtxt("./stock_data//SP500_clean_adapted.csv", delimiter=',', dtype=float))
        data_str = dc(np.genfromtxt("./stock_data//SP500_clean_adapted.csv", delimiter=',', dtype=str))
        data_num = dc(data_orig[:, 1])
        data_cal = list()
        # for i in range(len(orig_data_nasdaq_str)):
        #    nasdaq_x_date.append(datetime.strptime(orig_data_nasdaq_str[i,0], '%Y-%m-%d'))
        for iv in range(len(data_str)):
            data_cal.append(datetime.strptime(data_str[iv, 0], '%Y-%m-%d'))
        # nasdaq_x_count = np.array(range(len(orig_data_nasdaq_num)))
        data_count = dc(np.array(range(len(data_num))))
        x_SP = data_cal
        y_SP = data_num
    elif finance_data == 1:
        data_orig = dc(np.genfromtxt("./stock_data//dow_jones_clean_adapted.csv", delimiter=',', dtype=float))
        data_str = dc(np.genfromtxt("./stock_data//dow_jones_clean_adapted.csv", delimiter=',', dtype=str))
        data_num = dc(data_orig[:, 1])
        data_cal = list()
        # for i in range(len(orig_data_nasdaq_str)):
        #    nasdaq_x_date.append(datetime.strptime(orig_data_nasdaq_str[i,0], '%Y-%m-%d'))
        for iv in range(len(data_str)):
            data_cal.append(datetime.strptime(data_str[iv, 0], '%Y-%m-%d'))
        # nasdaq_x_count = np.array(range(len(orig_data_nasdaq_num)))
        data_count = dc(np.array(range(len(data_num))))
        x_DJ = data_cal
        y_DJ = data_num
    elif finance_data == 2:
        data_orig = dc(np.genfromtxt("./stock_data//nasdaq_clean_adapted.csv", delimiter=',', dtype=float))
        data_str = dc(np.genfromtxt("./stock_data//nasdaq_clean_adapted.csv", delimiter=',', dtype=str))
        data_num = dc(data_orig[:, 1])
        data_cal = list()
        # for i in range(len(orig_data_nasdaq_str)):
        #    nasdaq_x_date.append(datetime.strptime(orig_data_nasdaq_str[i,0], '%Y-%m-%d'))
        for iv in range(len(data_str)):
            data_cal.append(datetime.strptime(data_str[iv, 0], '%Y-%m-%d'))
        # nasdaq_x_count = np.array(range(len(orig_data_nasdaq_num)))
        data_count = dc(np.array(range(len(data_num))))

        x_NA = data_cal
        y_NA = data_num

    signal = dc(data_num[:max_len])
    print(len(signal))
    data_list_y = list()
    hurst_list = list()

    for ii in range(max_len):
        if ii + step_size > max_len: break
        Hursti, c, data = dc(hurst.compute_Hc(signal[ii:(ii+step_size)], simplified=False))
        data_list_y.append(data[-1])
        data_list_x = dc(np.array(data[0]))
        hurst_list.append(Hursti)
    av_log_data_finance[0,1:1+len(data_list_x)] = dc(data_list_x)
    std_log_data_finance[0,1:1+len(data_list_x)] = dc(data_list_x)
    hurst_av.append(np.mean(np.array(hurst_list)))
    hurst_std.append(np.std(np.array(hurst_list)))


    #print(data_list_y)
    data_list_y = dc(np.array(data_list_y))
    #print(data_list_y)
    #print('###################################')
    #print(data_list_y[:10,:10])
    for ii in range(len(data_list_x)):
        #print(data_list_y[:, ii])
        #sys.exit()
        av_log_data_finance[i+1, ii+1] = np.mean(data_list_y[:,ii])
        std_log_data_finance[i+1, ii+1] = np.std(data_list_y[:,ii])

    #print(data_list_x)
    #print(av_log_data_finance)
av_log_data_finance = dc(av_log_data_finance[:, :(1+len(data_list_x))])
std_log_data_finance = dc(std_log_data_finance[:, :(1+len(data_list_x))])

#print(av_log_data_finance)

if plot_assets:
    print('sdkj')
    # Create a new figure
    fig, ax = (plt.subplots(figsize=(10, 6)))

    # Plot the data
    x_SP = dc([mdates.date2num(d) for d in x_SP])

    ax.plot_date(x_SP, y_SP, '-k')
    print('######################')
    print(x_SP[:20])
    # Set the title
    ax.set_title('S&P500')

    # Set the x and y labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Show the plot
    #plt.tight_layout()
    plt.savefig('SP500_ts.png')
    plt.savefig('SP500_ts.eps')
    plt.show()

    # Create a new figure
    fig, ax = (plt.subplots(figsize=(10, 6)))

    # Plot the data
    x_DJ = dc([mdates.date2num(d) for d in x_DJ])

    ax.plot_date(x_DJ, y_DJ, data_num, '-k')
    print('######################')
    print(x_DJ[:20])

    # Set the title
    ax.set_title('Dow Jones')

    # Set the x and y labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Show the plot
    #plt.tight_layout()
    plt.savefig('DJ_ts.png')
    plt.savefig('DJ_ts.eps')
    plt.show()

    # Create a new figure
    x_NA = dc([mdates.date2num(d) for d in x_NA])

    fig, ax = (plt.subplots(figsize=(10, 6)))

    # Plot the data
    ax.plot_date(x_NA, y_NA, '-k')
    print('######################')
    print(x_NA[:20])
    # Set the title
    ax.set_title('NASDAQ')

    # Set the x and y labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Show the plot
    #plt.tight_layout()
    plt.savefig('NA_ts.png')
    plt.savefig('NA_ts.eps')
    plt.show()
    sys.exit()



av_log_data = np.zeros((len(alphas)+1,100))
av_log_data[1:,0] = dc(alphas)
std_log_data = np.zeros((len(alphas)+1,100))
std_log_data[1:,0] = dc(alphas)



for i in range(len(alphas)):
    #print(alphas[i])
    #signal = dc(flm.flm(alpha=alphas[i], H=0.5, n=9))
    if alphas[i] != 2.0:
        signal = dc(hurst.random_walk(12317, scaling_exponent))
    else:
        signal = dc(flm.flm(alpha=alphas[i], H=scaling_exponent, n=14))
        #signal = dc(flm.flm(alpha=alphas[i], H=0.5, n=9))


    signal = dc(signal[:max_len])
    #print(len(signal))
    data_list_y = list()

    for ii in range(max_len):
        if ii + step_size > max_len: break
        Hursti, c, data = dc(hurst.compute_Hc(signal[ii:(ii+step_size)], simplified=False))
        data_list_y.append(data[-1])
        data_list_x = dc(np.array(data[0]))
    av_log_data[0,1:1+len(data_list_x)] = dc(data_list_x)
    std_log_data[0,1:1+len(data_list_x)] = dc(data_list_x)

    #print(data_list_y)
    data_list_y = dc(np.array(data_list_y))
    #print(data_list_y)
    #print('###################################')
    #print(data_list_y[:10,:10])
    for ii in range(len(data_list_x)):
        #print(data_list_y[:, ii])
        #sys.exit()
        av_log_data[i+1, ii+1] = np.mean(data_list_y[:,ii])
        std_log_data[i+1, ii+1] = np.std(data_list_y[:,ii])

    #print(data_list_x)
    #print(av_log_data)
av_log_data = dc(av_log_data[:, :(1+len(data_list_x))])
std_log_data = dc(std_log_data[:, :(1+len(data_list_x))])


# Colors for alpha
colors_alpha = ['red', 'green', 'blue', 'purple']

# Colors for finance
colors_finance = ['cyan', 'magenta', 'orange']



# Assuming av_log_data is a numpy array and the alpha and scale values are defined as before
alpha_values = av_log_data[1:, 0]
scale_values = av_log_data[0, 1:]

alpha_values_finance = av_log_data_finance[1:, 0]
scale_values_finance = av_log_data_finance[0, 1:]

labels = ['alpha=0.5', 'alpha=1.0', 'alpha=1.5', 'fBm']

labels_finance = ['S&P500', 'Dow Jones', 'NASDAQ']


print('#############################################')
print('#############################################')
print('#############################################')
print('#############################################')
print('#############################################')
print(hurst_av)
print(hurst_std)
print('#############################################')
print('#############################################')
print('#############################################')
print('#############################################')
print('#############################################')
print('#############################################')
print('#############################################')






plt.figure()

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data.shape[0]):
    alpha = alpha_values[i - 1]
    values = av_log_data[i, 1:]

    # Plot on log-log scale with alpha as label
    plt.loglog(scale_values, values, label=labels[i-1])

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]

    # Plot on log-log scale with alpha as label
    plt.loglog(scale_values_finance, values_finance, label=labels_finance[i-1])

# Adding labels
plt.xlabel('τ')
plt.ylabel('R/S Ratio')
plt.title(f'Scaling Exponent: {scaling_exponent}')

# Adding legend
plt.legend()
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}.png')
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}.eps')



# Show the plot
plt.show()




# Set Seaborn style
sns.set_style("whitegrid")

# Assuming av_log_data is a numpy array and the alpha and scale values are defined as before
alpha_values = av_log_data[1:, 0]
scale_values = av_log_data[0, 1:]

alpha_values_finance = av_log_data_finance[1:, 0]
scale_values_finance = av_log_data_finance[0, 1:]

labels = ['fLm α=0.5', 'fLm α=1.0', 'fLm α=1.5', 'fLm fBm']
labels_finance = ['S&P500', 'Dow Jones', 'NASDAQ']

# Create a new figure
plt.figure(figsize=(10,6))

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data.shape[0]):
    alpha = alpha_values[i - 1]
    values = av_log_data[i, 1:]

    # Plot on log-log scale with alpha as label
    plt.loglog(scale_values, values, label=labels[i-1], linewidth=2)

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]

    # Plot on log-log scale with alpha as label
    plt.loglog(scale_values_finance, values_finance, label=labels_finance[i-1], linewidth=2)

# Adding labels
plt.xlabel('τ')
plt.ylabel('R/S Ratio')
plt.title(f'Scaling Exponent: {scaling_exponent}')

# Adding legend
plt.legend()

# Save the plot
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns.png')
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns.eps')

# Show the plot
plt.show()




# Set Seaborn style
sns.set_style("whitegrid")

# Assuming av_log_data is a numpy array and the alpha and scale values are defined as before
alpha_values = av_log_data[1:, 0]
scale_values = av_log_data[0, 1:]

alpha_values_finance = av_log_data_finance[1:, 0]
scale_values_finance = av_log_data_finance[0, 1:]

labels = ['fLm α=0.5', 'fLm α=1.0', 'fLm α=1.5', 'fBm']
labels_finance = ['S&P500', 'Dow Jones', 'NASDAQ']

# Line types for alpha
linetypes_alpha = ['-', '--', ':', '-.']

# Line types for finance
linetypes_finance = ['--', ':', '-.']

# Create a new figure
plt.figure(figsize=(10,6))

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data.shape[0]):
    alpha = alpha_values[i - 1]
    values = av_log_data[i, 1:]

    # Plot on log-log scale with alpha as label
    plt.loglog(scale_values, values, label=labels[i-1], linewidth=3, linestyle=linetypes_alpha[i-1])

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]

    # Plot on log-log scale with alpha as label
    plt.loglog(scale_values_finance, values_finance, label=labels_finance[i-1], linewidth=1.5, linestyle=linetypes_finance[i-1])

# Adding labels
plt.xlabel('τ')
plt.ylabel('R/S Ratio')
plt.title(f'Scaling Exponent: {scaling_exponent}')

# Adding legend
plt.legend()

# Save the plot
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns_2.png')
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns_2.eps')

# Show the plot
plt.show()





# Set Seaborn style
sns.set_style("whitegrid")

# Assuming av_log_data is a numpy array and the alpha and scale values are defined as before
alpha_values = av_log_data[1:, 0]
scale_values = av_log_data[0, 1:]

alpha_values_finance = av_log_data_finance[1:, 0]
scale_values_finance = av_log_data_finance[0, 1:]

labels = ['fLm α=0.5', 'fLm α=1.0', 'fLm α=1.5', 'fBm']
labels_finance = ['S&P500', 'Dow Jones', 'NASDAQ']

# Line types for alpha
linetypes_alpha = ['-', '--', ':', '-.']

# Line types for finance
linetypes_finance = ['--', ':', '-.']

# Create a new figure
fig, ax = plt.subplots(figsize=(20,12))

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data.shape[0]):
    alpha = alpha_values[i - 1]
    values = av_log_data[i, 1:]

    # Plot on log-log scale with alpha as label
    ax.loglog(scale_values, values, label=labels[i-1], linewidth=3, linestyle=linetypes_alpha[i-1], color=colors_alpha[i-1])

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]

    # Plot on log-log scale with alpha as label
    ax.loglog(scale_values_finance, values_finance, label=labels_finance[i-1], linewidth=1.5, linestyle=linetypes_finance[i-1], color=colors_finance[i-1])

# Adding labels and title with increased size
ax.set_xlabel('τ', fontsize=20)
ax.set_ylabel(r'$\frac{\overline{R}}{\overline{S}}$', rotation=0, fontsize=20, labelpad=30)  # Rotated and larger ylabel
ax.set_title(f'Scaling Exponent: {scaling_exponent}', fontsize=20)

# Create an inset_axes instance with a width of 30% and a height of 40% of the parent axes' bounding box
#axins = inset_axes(ax, width="30%", height="40%", loc='upper left')
axins = inset_axes(ax, width="28%", height="37.5%", bbox_to_anchor=(0.055, 0.61, 1, 1), bbox_transform=ax.transAxes, loc=3)


# Zoom into the last 10 data points
zoom_start = len(scale_values) - 2

# Loop for plotting in the zoomed area
for i in range(1, av_log_data.shape[0]):
    if i == 4:
        print('out')
    else:
        values = av_log_data[i, 1:]
        axins.loglog(scale_values[zoom_start:], values[zoom_start:], linewidth=3, linestyle=linetypes_alpha[i-1], color=colors_alpha[i-1])

for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]
    axins.loglog(scale_values_finance[zoom_start:], values_finance[zoom_start:], linewidth=1.5, linestyle=linetypes_finance[i-1], color=colors_finance[i-1])


# Remove y-axis labels and ticks of the inset
# Remove y-axis labels and ticks of the inset
axins.yaxis.set_ticks([])  # This line removes the ticks
# Limit the x-axis to 2 ticks
#axins.set_xticks([scale_values_finance[zoom_start]+2, scale_values_finance[-1]])
axins.set_xticklabels([scale_values_finance[zoom_start]+2, scale_values_finance[-1]], rotation=45)

# Adding legend
ax.legend(loc='lower right')

# Save the plot
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns_2_zoom1.png')
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns_2_zoom1.eps')

# Show the plot
plt.show()








# Set Seaborn style
sns.set_style("whitegrid")

# Assuming av_log_data is a numpy array and the alpha and scale values are defined as before
alpha_values = av_log_data[1:, 0]
scale_values = av_log_data[0, 1:]

alpha_values_finance = av_log_data_finance[1:, 0]
scale_values_finance = av_log_data_finance[0, 1:]

labels = ['fLm α=0.5', 'fLm α=1.0', 'fLm α=1.5', 'fBm']
labels_finance = ['S&P500', 'Dow Jones', 'NASDAQ']

# Line types for alpha
linetypes_alpha = ['-', '--', ':', '-.']

# Line types for finance
linetypes_finance = ['--', ':', '-.']

# Create a new figure
fig, ax = plt.subplots(figsize=(20,12))

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data.shape[0]):
    alpha = alpha_values[i - 1]
    values = av_log_data[i, 1:]
    ax.loglog(scale_values, values, label=labels[i-1], linewidth=3, linestyle=linetypes_alpha[i-1])

# Loop through each row (excluding the first row of scales)
for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]
    ax.loglog(scale_values_finance, values_finance, label=labels_finance[i-1], linewidth=1.5, linestyle=linetypes_finance[i-1])

# Adding labels and title with increased size
ax.set_xlabel('τ', fontsize=20)
ax.set_ylabel(r'$\frac{\overline{R}}{\overline{S}}$', rotation=0, fontsize=20, labelpad=30)  # Rotated and larger ylabel
ax.set_title(f'Scaling Exponent: {scaling_exponent}', fontsize=20)

# Create an inset_axes instance with a width of 30% and a height of 40% of the parent axes' bounding box
axins = inset_axes(ax, width="30%", height="40%", loc='lower right')

# Zoom into the last 10% data points
zoom_start = int(len(scale_values) * 0.9)

# Loop for plotting in the zoomed area
for i in range(1, av_log_data.shape[0]):
    values = av_log_data[i, 1:]
    axins.loglog(scale_values[zoom_start:], values[zoom_start:], linewidth=3, linestyle=linetypes_alpha[i-1])

for i in range(1, av_log_data_finance.shape[0]):
    values_finance = av_log_data_finance[i, 1:]
    axins.loglog(scale_values_finance[zoom_start:], values_finance[zoom_start:], linewidth=1.5, linestyle=linetypes_finance[i-1])

# Remove y-axis labels and ticks of the inset
axins.set_yticks([])

# Limit the x-axis to 2 ticks
axins.set_xticks([scale_values_finance[zoom_start], scale_values_finance[-1]])

# Adding legend
ax.legend(loc='lower right', fontsize=20)

# Save the plot
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns_2_zoom0.png')
plt.savefig(f'2d_plot_fLm_finance_{scaling_exponent}_sns_2_zoom0.eps')

# Show the plot
plt.show()







