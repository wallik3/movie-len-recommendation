from functools import partial
import torch
import torch.nn.functional as F
from typing import Union, Callable, List, Literal
import pandas as pd
from movie_len_recommendation.constant import (
    normalized_movie_mapping,
    normalized_user_mapping,
    unnormalized_movie_mapping,
    unnormalized_user_mapping
)

def get_similar_items(
    user_id:int,
    k:int,
    embeddings,
    normalized_item_mapping,
    unnormalized_item_mapping,
    is_input_normalized:bool=True,
    is_output_unnormalized=True,
    ):
  """
  Get the similar user based on the cosine similarity

  Args
  ----
  is_input_normalized (bool) : is the input_user is normalized or not
  is_input_normalized (bool) : is the user is normalized or not
  """
  normalized_user_id = normalized_item_mapping[user_id] if not is_input_normalized else user_id

  vector = embeddings[normalized_user_id]
  matrix = embeddings
  # Expand the vector to match the matrix's shape
  vector_expanded = vector.unsqueeze(0).expand_as(matrix)

  cosine_score = F.cosine_similarity(matrix, vector_expanded, dim=1)

  topk_values, topk_indices = torch.topk(cosine_score, k+1)

  # Pop the first index of both tensors by slicing
  topk_values_popped = topk_values[1:].tolist()
  topk_indices_popped = topk_indices[1:].tolist()

  if is_output_unnormalized:
    topk_indices_popped = [unnormalized_item_mapping[id] for id in topk_indices_popped]

  return topk_values_popped, topk_indices_popped

# Partialize the function
get_similar_users = partial(get_similar_items, normalized_item_mapping=normalized_user_mapping, unnormalized_item_mapping=unnormalized_user_mapping)
get_similar_movies = partial(get_similar_items, normalized_item_mapping=normalized_movie_mapping, unnormalized_item_mapping=unnormalized_movie_mapping)

def get_user_watched_movie_id(user_id:Union[int,List[int]])->List[int]:
  """
  Get the watched movies from user id
  """
  global rating
  watched_movies = ratings.query(f"userId == {user_id}") if isinstance(user_id,int) else ratings.query(f"userId in @user_id")
  watched_movies = list(watched_movies["movieId"].unique())
  return watched_movies

def get_movie_candidate_id(
    user_id_list:List[int],
    exclude_movie_id_list:List[int],
    k:int = 5
    )->List[int]:
  global ratings
  candidates = get_user_watched_movie_id(user_id_list)

  movie_candidate_id_list = set(candidates) - set(exclude_movie_id_list)
  return movie_candidate_id_list

def recommend_by_similar_score(
    user_id:int,
    movie_candidate_id_list:List[int],
    model:Callable,
    device,
    k:int = 5,
    is_input_normalized:bool=True,
    is_output_unnormalized:bool=True
    ):
  """
    Recommend the user based on the predicted rating score between the user and all movie's candidate
  """
  global ratings, normalized_user_mapping, unnormalized_user_mapping, normalized_movie_mapping, unnormalized_movie_mapping

  normalized_user_id = normalized_user_mapping[user_id] if is_input_normalized else user_id
  normalized_movie_id_list = [normalized_movie_mapping[movie_id] for movie_id in movie_candidate_id_list]

  normalized_user_id_tensor = torch.tensor([normalized_user_id],dtype=torch.long).to(device)
  normalized_movie_id_tensor = torch.tensor(normalized_movie_id_list,dtype=torch.long).to(device)

  # Inference the rating
  pred_ratings = model(normalized_user_id_tensor,normalized_movie_id_tensor)

    # Use torch.topk to get the top k values and their indices
  top_score, top_movie = torch.topk(pred_ratings,k)

  # Convert the results to CPU if needed for easier readability
  top_score = list(top_score.cpu().detach().numpy())
  top_movie = list(top_movie.cpu().detach().numpy())

  # TODO : output normalize back
  top_movie = [unnormalized_movie_mapping[movie_id] for movie_id in top_movie]

  return top_score, top_movie


# Pre calculate for weighted_average_rating method
# Function to calculate similarity matrix (using Pearson correlation for this example)
def recommend(user_id,
              user_embeddings,
              model,
              device,
              k:int=5):
  """
  Finalize the function
  """
  # Exclude the watched movie from a list of recommended item
  excluded_movie_candidates = get_user_watched_movie_id(user_id)

  # Get the most similar user using user_embeddings
  _, top_user_id_list = get_similar_users(user_id,k,user_embeddings)

  # Get all movie candidates from the similar user
  movie_candidate_id_list = get_movie_candidate_id(top_user_id_list, excluded_movie_candidates)

  # Get Top movie with their score for recommending
  top_score, top_movie = recommend_by_similar_score(user_id, movie_candidate_id_list, model, device, k)

  return top_score, top_movie