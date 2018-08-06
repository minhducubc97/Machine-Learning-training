##### PYTHON 2 AND 3 SUPPORT
from __future__ import division, print_function, unicode_literals

##### COMMON IMPORT
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import warnings

##### INITIALIZATION
# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
# %matplotlib - Not available in VSCode, only in IPython API
plt.ion()

# Specify sizes
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

##### HELPER FUNCTIONS
# Where to save the figures
# PROJECT_ROOT_DIR = "."
# def save_fig(fig_id, tight_layout=True):
#     path = os.path.join(PROJECT_ROOT_DIR, "graphs", fig_id + ".png")
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Merge the OECD's life satisfaction data and the IMF's GDP per capita data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

##### MAIN CODE
# Load the data
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is the Project Root
oecd_bli = pd.read_csv(ROOT_DIR + "\datasets\lifesat\oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(ROOT_DIR + "\datasets\lifesat\gdp_per_capita.csv", thousands=',',delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
# np.c_ : create a 2D array
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model - Linear Regression
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capital
print(model.predict(X_new)) # outputs [[ 5.96242338]]]

# NOTE: Run the file by typing python -i linear_model.py into the terminal
