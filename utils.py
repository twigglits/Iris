import pandas as pd
import numpy as np
import sklearn.datasets as datasets
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def load_iris_data():
    logger.info("Loading iris data with species.")
    iris = datasets.load_iris()
    df = (
        pd.DataFrame(
            data=np.c_[iris["data"], iris["target"]],
            columns=iris["feature_names"] + ["target"],
        )
        .astype({"target": int})
        .assign(
            species=lambda x: x["target"].map(dict(enumerate(iris["target_names"])))
        )
    )
    logger.success("Done: Iris data loaded.")
    return df
