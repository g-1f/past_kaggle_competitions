import os
import cv2
import pydicom
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Input,
    BatchNormalization,
    GlobalAveragePooling2D,
    Add,
    Conv2D,
    AveragePooling2D,
    LeakyReLU,
    Concatenate,
)
from tensorflow.keras import Model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
import tensorflow.keras.applications as tfa
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

EPOCHS = 2
BATCH_SIZE = 8
NFOLD = 5
LR = 0.003
SAVE_BEST = True
MODEL_CLASS = "b1"

train = pd.read_csv("../input/osic-pulmonary-fibrosis-progression/train.csv")


def get_tab(df):
    vector = [(df.Age.values[0] - 30) / 30]

    if df.Sex.values[0].lower() == "male":
        vector.append(0)
    else:
        vector.append(1)

    if df.SmokingStatus.values[0] == "Never smoked":
        vector.extend([0, 0])
    elif df.SmokingStatus.values[0] == "Ex-smoker":
        vector.extend([1, 1])
    elif df.SmokingStatus.values[0] == "Currently smokes":
        vector.extend([0, 1])
    else:
        vector.extend([1, 0])
    return np.array(vector)


A = {}
TAB = {}
P = []
for i, p in tqdm(enumerate(train.Patient.unique())):
    sub = train.loc[train.Patient == p, :]
    fvc = sub.FVC.values
    weeks = sub.Weeks.values
    c = np.vstack([weeks, np.ones(len(weeks))]).T
    a, b = np.linalg.lstsq(c, fvc)[0]

    A[p] = a
    TAB[p] = get_tab(sub)
    P.append(p)


def get_img(path):
    d = pydicom.dcmread(path)
    return cv2.resize(
        (d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (512, 512)
    )

    x, y = [], []


for p in tqdm(train.Patient.unique()):
    try:
        ldir = os.listdir(
            f"../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_noise/mask_noise/{p}/"
        )
        numb = [float(i[:-4]) for i in ldir]
        for i in ldir:
            x.append(
                cv2.imread(
                    f"../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_noise/mask_noise/{p}/{i}",
                    0,
                ).mean()
            )
            y.append(float(i[:-4]) / max(numb))
    except:
        pass
