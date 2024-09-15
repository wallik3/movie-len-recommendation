import pandas as pd
import numpy as np
from movie_len_recommendation.utils.dtype import (
    get_normalize_x, 
    reverse_dictionary
)
from movie_len_recommendation.dataset import get_dataset

ratings = get_dataset("ratings")
# Make normalized and unnormalized mapping for both user and movie
normalized_movie_mapping = get_normalize_x(list(np.sort(ratings.movieId.unique())))
normalized_user_mapping = get_normalize_x(list(np.sort(ratings.userId.unique())))

# Again, we make a reverse dictionary to map back
unnormalized_movie_mapping = reverse_dictionary( normalized_movie_mapping )
unnormalized_user_mapping = reverse_dictionary( normalized_user_mapping )