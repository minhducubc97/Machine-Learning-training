#===================================== COMMON SETUP =====================================
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
# to make the output stable across many runs (different seeds lead to different result)
np.random.seed(42)

# To plot pretty figures
# %matplotlib - Not available in VSCode, only in IPython API
plt.ion()

# Specify sizes
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

##### COMMON HELPER FUNCTIONS
# Where to save the figures
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(ROOT_DIR, "graphs", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
#===================================== COMMON SETUP =====================================

##### NEW IMPORT
import tarfile
from six.moves import urllib
import pandas as pd

##### CONSTANTS
DOWNLOAD_LINK = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is the Project Root
HOUSING_PATH = ROOT_DIR + "\\datasets\\housing\\"

##### HELPER FUNCTIONS
# Get the housing data from the Url
def fetch_housing_data(housing_url = DOWNLOAD_LINK, housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    path_to_tgz = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, path_to_tgz)
    housing_tgz = tarfile.open(path_to_tgz)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path = HOUSING_PATH):
    path_to_csv = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(path_to_csv)

#fetch_housing_data() # get the data from the given URL
housing = load_housing_data()
#housing.head() # display the table in csv file
#housing.describe() #show mean, average, min, ... all the values arranged
housing.hist(bins=50, figsize=(20,15))
save_fig("histograms")
plt.show()