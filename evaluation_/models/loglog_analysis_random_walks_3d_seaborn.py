
import hurst

from copy import deepcopy as dc

import flm


import numpy as np




alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

#alphas = [0.1, 0.2, 0.3]


scaling_exponent = 0.5

max_len = 12317
#max_len = 178
step_size = 1000
#step_size = 150


av_log_data = np.zeros((len(alphas)+1,100))
av_log_data[1:,0] = dc(alphas)
std_log_data = np.zeros((len(alphas)+1,100))
std_log_data[1:,0] = dc(alphas)

for i in range(len(alphas)):
    print(alphas[i])
    #signal = dc(flm.flm(alpha=alphas[i], H=0.5, n=9))
    signal = dc(flm.flm(alpha=alphas[i], H=scaling_exponent, n=14))
    #signal = dc(hurst.random_walk(12317, 0.5))

    signal = dc(signal[:max_len])
    print(len(signal))
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
    print(data_list_y)
    print('###################################')
    print(data_list_y[:10,:10])
    for ii in range(len(data_list_x)):
        #print(data_list_y[:, ii])
        #sys.exit()
        av_log_data[i+1, ii+1] = np.mean(data_list_y[:,ii])
        std_log_data[i+1, ii+1] = np.std(data_list_y[:,ii])

    print(data_list_x)
    print(av_log_data)
av_log_data = dc(av_log_data[:, :(1+len(data_list_x))])
std_log_data = dc(std_log_data[:, :(1+len(data_list_x))])

print(av_log_data)
























import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set Seaborn style
sns.set_style("whitegrid")

# Create a meshgrid for alpha and scale values
alpha_values = av_log_data[1:, 0]
scale_values = av_log_data[0, 1:]
X, Y = np.meshgrid(alpha_values, np.log10(scale_values))

# Plot using 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface, converting Z-axis to log scale
Z = np.log10(av_log_data[1:, 1:])
Z = np.transpose(Z)  # Transpose Z

# Now plot
ax.plot_surface(X, Y, Z, cmap='viridis')

# Adding labels
ax.set_xlabel('Alpha')
ax.set_ylabel('log(τ)')
ax.set_zlabel('log(R/S Ratio)')
ax.set_title(f'Scaling Exponent: {scaling_exponent}')
plt.tight_layout()

# Save the plot
plt.savefig(f'3d_plot_fLm_{scaling_exponent}_{step_size}.png')
plt.savefig(f'3d_plot_fLm_{scaling_exponent}_{step_size}.eps')

# Show the plot
plt.show()

print(scaling_exponent)



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define colormap
cmap = plt.get_cmap("viridis")

# Set seaborn aesthetic style
sns.set_style("whitegrid")

# Create a 2D plot
fig, ax = plt.subplots()

# Plot curves with alternating colours and line styles
line_styles = ['-', '--', ':', '-.']
for i in range(len(alphas)):
    line_color = cmap(i / len(alphas))  # colour according to alpha value
    line_style = line_styles[i % len(line_styles)]  # cycle through line styles
    ax.plot(np.log10(scale_values), np.log10(av_log_data[i+1, 1:]),
            label=f'fLm α={alphas[i]}', color=line_color, linestyle=line_style)

ax.set_xlabel('log(τ)')
ax.set_ylabel('log(R/S Ratio)')
plt.title(f'Scaling Exponent: {scaling_exponent}')
plt.tight_layout()

# Save the plot
plt.savefig(f'2d_plot_fLm_{scaling_exponent}_{step_size}_sns.png')
plt.savefig(f'2d_plot_fLm_{scaling_exponent}_{step_size}_sns.eps')


# Draw and save the legend separately
fig_leg = plt.figure(figsize=(1.4, 4.3))
ax_leg = fig_leg.add_subplot(111)

# Add the legend from the previous axis
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')

# Hide the axes frame and the x/y labels
ax_leg.axis('off')
plt.tight_layout()

fig_leg.savefig(f'legend_{step_size}_sns.png')
fig_leg.savefig(f'legend_{step_size}_sns.eps')

# Show the plot
plt.show()
