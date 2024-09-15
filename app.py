import os
import pandas as pd
import torch
from flask import Flask, request, jsonify
from movie_len_recommendation.recommend import recommend
from movie_len_recommendation.dataset import get_dataset
from movie_len_recommendation.model import get_mf_architecture

app = Flask(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import dataset (Unextract the file first)
movies = get_dataset("movies")
ratings = get_dataset("ratings")

# Get the best atchitecture
num_users, num_items = ratings.userId.nunique(), ratings.movieId.nunique()
architecture = get_mf_architecture("default")

# Load the best parameters to 
model = architecture(num_users, num_items, embedding_size=100).to(DEVICE)
model.load_state_dict(
    torch.load(
        os.path.normpath('artifact/model/best_mfnet_param.pth'),
        weights_only=True
    )
)
user_embeddings = model.user_embeddings.weight.data

@app.route('/recommend_movie', methods=['POST'])
def recommend_movie():
    data = request.get_json()
    user_id = data.get('user_id')
    _, top_movie = recommend(user_id,user_embeddings,model,DEVICE)
    top_movie_name = list(movies.query("movieId in @top_movie").title.unique())
    response = {
        "recommended_movie" : top_movie_name
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
