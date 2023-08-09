import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import seaborn as sns
import numpy as np

# Path to your csv file
#file_path = './COMP_DATA/out_finance__200_50_MLP_LightGBM_CatBoost_AdaBoost_Lasso_Ridge_SP500.csv'
#file_path = './COMP_DATA/out_finance__200_50_MLP_LightGBM_CatBoost_AdaBoost_Lasso_Ridge_DowJones.csv'
#file_path = './COMP_DATA/out_finance__200_50_MLP_LightGBM_CatBoost_AdaBoost_Lasso_Ridge_Nasdaq.csv'
#file_path = './COMP_DATA/out_finance__200_50_MLP_CatBoost.csv'
#file_path = './COMP_DATA/out_finance__350_10_MLP_CatBoost.csv'
#file_path = './COMP_DATA/out_finance__200_10_MLP_CatBoost.csv'
#file_path = './COMP_DATA/out_finance__200_10_MLP_CatBoost_ad_Nasdaq.csv'
#file_path = './COMP_DATA/out_finance__200_10_MLP_CatBoost_ad_SP500.csv'
#file_path = './COMP_DATA/out_finance__200_10_MLP_CatBoost_ad_DowJones.csv'
#file_path = './COMP_DATA/out_finance__200_50_MLP_CatBoost_ad_Nasdaq.csv'
#file_path = './COMP_DATA/out_finance__200_50_MLP_CatBoost_ad_SP500.csv'
#file_path = './COMP_DATA/out_finance__200_50_MLP_CatBoost_ad_DowJones.csv'
#file_path = './COMP_DATA/out_finance__350_10_MLP_CatBoost_ad_Nasdaq.csv'
#file_path = './COMP_DATA/out_finance__350_10_MLP_CatBoost_ad_SP500.csv'
#file_path = './COMP_DATA/out_finance__350_10_MLP_CatBoost_ad_DowJones.csv'
#file_path = './COMP_DATA/out_finance__350_50_MLP_CatBoost_ad_Nasdaq.csv'
#file_path = './COMP_DATA/out_finance__350_50_MLP_CatBoost_ad_SP500.csv'
file_path = './COMP_DATA/out_finance__350_50_MLP_CatBoost_ad_DowJones.csv'


#file_path = './COMP_DATA/out_finance__350_10_MLP_CatBoost.csv'



# Save the plots in the specified folder
output_folder = "./separate_plots_dfa/"
os.makedirs(output_folder, exist_ok=True)

# Extract information from file name
file_name = os.path.basename(file_path)  # get the file name
file_name_no_ext = os.path.splitext(file_name)[0]  # remove the extension
info = file_name_no_ext.split('_')  # split on underscore

asset_name = info[-1]  # get the last word
sliding_window = info[3]  # get the sliding window value
step_size = info[4]  # get the step size value

# Load your csv file
df = pd.read_csv(file_path)

# Replace '-' in column names with '_'
#df.columns = df.columns.str.replace('-', '_')

# Convert your second column to datetime format
df.iloc[:, 1] = pd.to_datetime(df.iloc[:, 1])

# Replace '-' with '_' in column names
df.columns = df.columns.str.replace('-', '_')

# Define the line styles
styles = ['-', '--', '-.', ':']

for i in range(3, df.shape[1]): # Iterating through all columns starting from the sixth one

    # Plot all other columns against the datetime column
    fig, ax = plt.subplots()

    ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=1.2, label=df.columns[5])
    ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.7, label=df.columns[i])

    ax.legend(loc='upper left')

    fig.tight_layout()

    # Set x-axis major locator to auto
    locator = mdates.AutoDateLocator()
    formatter = mdates.AutoDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    # Rotate date labels to prevent overlap
    fig.autofmt_xdate()

    # Adjust the subplot to prevent cut-off labels
    plt.subplots_adjust(bottom=0.2)

    # Save the plots in the specified folder
    #plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_{df.columns[i]}.png')
    #plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_{df.columns[i]}.eps')

    # Close the plot to free up memory
    plt.close(fig)

# Compute the correlation matrix
#corr = df.iloc[:, 5:].corr()
corr = df.iloc[:, 3:].corr()


# Generate a mask for the upper triangle
#mask = np.triu(np.ones_like(corr, dtype=bool))

# Plot the heatmap
plt.figure(figsize=(20, 17))
cmap = sns.diverging_palette(230, 20, as_cmap=True)  # create a custom palette
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, cbar_kws={"shrink": .5},
            xticklabels=True, yticklabels=True)

#plt.yticks(fontsize=14)  # Adjust to your desired font size
plt.yticks(rotation=0, fontsize=14)  # Adjust to your desired font size and rotate labels

plt.xticks(fontsize=14)  # Adjust to your desired font size

plt.subplots_adjust(bottom=0.2, left=0.2) # adjust the distance of labels
plt.title(f'Correlations - {asset_name} - {sliding_window} - {step_size}')
plt.tight_layout()
plt.savefig(f'{output_folder}correlation_heatmap_{asset_name}_{sliding_window}_{step_size}.png')
plt.savefig(f'{output_folder}correlation_heatmap_{asset_name}_{sliding_window}_{step_size}.eps')
plt.close()

# Save the correlation matrix as a LaTeX table
with open(f'{output_folder}correlation_matrix.tex', 'w') as f:
    f.write(corr.to_latex())

# Find the top 5 off-diagonal correlations
correlations = corr.unstack().sort_values(ascending=False)
top_5_correlations = correlations[correlations.index.get_level_values(0) != correlations.index.get_level_values(1)][:5]

print('Top 5 correlations:')
for idx, value in top_5_correlations.items():
    print(f'{idx[0]} and {idx[1]}: {value}')

line_styles = ['-', '--']


# Other part of your code remains unchanged

# Define the line styles
styles = ['-', '--', '-.', ':']

# Plot all other columns against the datetime column

fig, ax = plt.subplots(figsize=(15, 7.5))

# Loop through the columns starting from 6th column
for i in range(3, df.shape[1]):
    if i != 5:
        line_style = line_styles[i % len(line_styles)]  # cycle through line styles
        ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.9, label=df.columns[i], linestyle=line_style)

ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=2, label=df.columns[5], color='black')


# Add the legend and set its location
ax.legend(loc='lower left')
plt.title(f'{asset_name} - {sliding_window} - {step_size}')

fig.tight_layout()

# Set x-axis major locator to auto
locator = mdates.AutoDateLocator()
formatter = mdates.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate date labels to prevent overlap
fig.autofmt_xdate()

# Adjust the subplot to prevent cut-off labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()


# Save the plots in the specified folder
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_all_models.png')
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_all_models.eps')

# Close the plot to free up memory
plt.close(fig)

# The rest of your code remains unchanged


# Other part of your code remains unchanged

# Define the line styles
styles = ['-', '--', '-.', ':']

# Plot all other columns against the datetime column

fig, ax = plt.subplots(figsize=(15, 7.5))

# Loop through the columns starting from 6th column
ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=3, label=df.columns[5], color='grey')

for i in range(3, 6):
    if i != 5:
        line_style = line_styles[i % len(line_styles)]  # cycle through line styles
        ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.9, label=df.columns[i], linestyle=line_style)

# Add the legend and set its location
ax.legend(loc='lower left')
plt.title(f'{asset_name} - {sliding_window} - {step_size}')
fig.tight_layout()

# Set x-axis major locator to auto
locator = mdates.AutoDateLocator()
formatter = mdates.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate date labels to prevent overlap
fig.autofmt_xdate()

# Adjust the subplot to prevent cut-off labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()

# Save the plots in the specified folder
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_only_algorithms.png')
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_only_algorithms.eps')

# Close the plot to free up memory
plt.close(fig)

# The rest of your code remains unchanged






# Other part of your code remains unchanged

# Define the line styles

# Plot all other columns against the datetime column

fig, ax = plt.subplots(figsize=(15, 7.5))


ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=3, label=df.columns[5], color='grey')
# Loop through the columns starting from 6th column
for i in range(5, df.shape[1]):
    if i != 5:
        line_style = line_styles[i % len(line_styles)]  # cycle through line styles
        ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.9, label=df.columns[i], linestyle=line_style)

# Add the legend and set its location
ax.legend(loc='lower left')
plt.title(f'{asset_name} - {sliding_window} - {step_size}')

fig.tight_layout()

# Set x-axis major locator to auto
locator = mdates.AutoDateLocator()
formatter = mdates.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate date labels to prevent overlap
fig.autofmt_xdate()

# Adjust the subplot to prevent cut-off labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
# Save the plots in the specified folder
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_mlmodels.png')
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_mlmodels.eps')

# Close the plot to free up memory
plt.close(fig)

# The rest of your code remains unchanged









# Other part of your code remains unchanged

# Define the line styles

# Filter the DataFrame to include only data from 1960 to 1980
df = df[(df.iloc[:, 1].dt.year >= 1960) & (df.iloc[:, 1].dt.year <= 1980)]

# Plot all other columns against the datetime column

fig, ax = plt.subplots(figsize=(15, 7.5))

# Loop through the columns starting from 6th column
ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=3, label=df.columns[5], color='grey')

for i in range(5, df.shape[1]):
    if i != 5:
        line_style = line_styles[i % len(line_styles)]  # cycle through line styles
        ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.9, label=df.columns[i], linestyle=line_style)

# Add the legend and set its location
ax.legend(loc='lower left')
plt.title(f'{asset_name} - {sliding_window} - {step_size}')

fig.tight_layout()

# Set x-axis major locator to auto
locator = mdates.AutoDateLocator()
formatter = mdates.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate date labels to prevent overlap
fig.autofmt_xdate()

# Adjust the subplot to prevent cut-off labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
# Save the plots in the specified folder
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_mlmodels19601980.png')
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_mlmodels19601980.eps')

# Close the plot to free up memory
plt.close(fig)

# The rest of your code remains unchanged




# Plot all other columns against the datetime column

fig, ax = plt.subplots(figsize=(15, 7.5))
ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=3, label=df.columns[5], color='grey')

# Loop through the columns starting from 6th column
for i in range(3, 6):
    if i != 5:
        line_style = line_styles[i % len(line_styles)]  # cycle through line styles
        ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.9, label=df.columns[i], linestyle=line_style)

# Add the legend and set its location
ax.legend(loc='lower left')
plt.title(f'{asset_name} - {sliding_window} - {step_size}')
fig.tight_layout()

# Set x-axis major locator to auto
locator = mdates.AutoDateLocator()
formatter = mdates.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate date labels to prevent overlap
fig.autofmt_xdate()

# Adjust the subplot to prevent cut-off labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()

# Save the plots in the specified folder
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_only_algorithms19601980.png')
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_only_algorithms19601980.eps')

# Close the plot to free up memory
plt.close(fig)


fig, ax = plt.subplots(figsize=(15, 7.5))



# Loop through the columns starting from 6th column
for i in range(3, df.shape[1]):
    if i != 5:
        line_style = line_styles[i % len(line_styles)]  # cycle through line styles
        ax.plot(df.iloc[:, 1], df.iloc[:, i], linewidth=0.9, label=df.columns[i], linestyle=line_style)

ax.plot(df.iloc[:, 1], df.iloc[:, 5], linewidth=3, label=df.columns[5], color='black')


# Add the legend and set its location
ax.legend(loc='lower left')
plt.title(f'{asset_name} - {sliding_window} - {step_size}')

fig.tight_layout()

# Set x-axis major locator to auto
locator = mdates.AutoDateLocator()
formatter = mdates.AutoDateFormatter(locator)

ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Rotate date labels to prevent overlap
fig.autofmt_xdate()

# Adjust the subplot to prevent cut-off labels
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()


# Save the plots in the specified folder
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_all_models19601980.png')
plt.savefig(f'{output_folder}plot_{asset_name}_{sliding_window}_{step_size}_dfa_all_models19601980.eps')

# Close the plot to free up memory
plt.close(fig)


