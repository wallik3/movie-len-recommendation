# MovieLens Recommendation System
## About
- Our goal is to recommend movies to users, a crucial task in the industry for upselling and other purposes. 
- In this project, we focus on the MovieLens platform. To achieve this, we will implement a machine learning framework to develop a recommendation system. 
- We will use the commonly employed technique of matrix factorization and demonstrate the results through a user collaborative learning framework.
- We developed the recommendation system in `notebooks/1_modelling.ipynb`.
- Finally, The model parameter can be found at `artifact/model/best_mfnet_param.pth`.
- To start the API server, run make run-server. This will enable the recommend_movie route. To get a recommended movie name, send a POST request with the following format: {user_id: int}.
