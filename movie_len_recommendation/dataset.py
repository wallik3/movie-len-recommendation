import pandas as pd
import os
from typing import Literal
from functools import lru_cache

@lru_cache(maxsize=None)
def get_dataset(name:Literal["movies","ratings","links","tags"])->pd.DataFrame: 
    """
    Get the dataset from the given name
    """
    file_mapping = {
        "movies" : "./artifact/dataset/movies.csv",
        "ratings" : "./artifact/dataset/ratings.csv",
        "links" : "./artifact/dataset/links.csv",
        "tags" : "./artifact/dataset/tags.csv",
    }
    filename = os.path.normpath(file_mapping[name])
    dataset = pd.read_csv(filename)
    return dataset