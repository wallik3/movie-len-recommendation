# MovieLens Recommendation System
## About
- Our goal is to recommend movies to users, a crucial task in the industry for upselling and other purposes. 
- In this project, we focus on the MovieLens platform. To achieve this, we will implement a machine learning framework to develop a recommendation system. 
- We will use the commonly employed technique of matrix factorization and demonstrate the results through a user collaborative learning framework.
- We developed the recommendation system in `notebooks/1_modelling.ipynb`.
- Finally, The model parameter can be found at `artifact/model/best_mfnet_param.pth`.
- To start the API server, run make run-server. This will enable the recommend_movie route. To get a recommended movie name, send a POST request with the following format: {user_id: int}.

## Demonstrate Example
- We demonstrate making a recommendation for user ID 9, who has once watched the following movie.

| movieId | title                                                    | genres                              |
|---------|----------------------------------------------------------|-------------------------------------|
| 37      | Richard III (1995)                                       | Drama|War                           |
| 158     | Party Girl (1995)                                        | Comedy                              |
| 190     | Clerks (1994)                                            | Comedy                              |
| 329     | Paper, The (1994)                                        | Comedy|Drama                        |
| 532     | Last Supper, The (1995)                                  | Drama|Thriller                      |
| 704     | Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)            | Drama|Film-Noir|Romance             |
| 705     | Citizen Kane (1941)                                      | Drama|Mystery                       |
| 794     | Lawnmower Man, The (1992)                                | Action|Horror|Sci-Fi|Thriller       |
| 834     | Glengarry Glen Ross (1992)                               | Drama                               |
| 900     | Raiders of the Lost Ark (Indiana Jones and the...)       | Action|Adventure                    |
| 969     | Back to the Future (1985)                                | Adventure|Comedy|Sci-Fi             |
| 1259    | Witness (1985)                                           | Drama|Romance|Thriller              |
| 1464    | Prom Night (1980)                                        | Horror                              |
| 1486    | Back to the Future Part II (1989)                        | Adventure|Comedy|Sci-Fi             |
| 1487    | Back to the Future Part III (1990)                       | Adventure|Comedy|Sci-Fi|Western     |
| 1498    | Godfather: Part III, The (1990)                          | Crime|Drama|Mystery|Thriller        |
| 1711    | Producers, The (1968)                                    | Comedy                              |
| 2161    | Tommy (1975)                                             | Musical                             |
| 2184    | Phantasm (1979)                                          | Horror|Sci-Fi                       |
| 2391    | Any Given Sunday (1999)                                  | Drama                               |
| 2494    | Ghost Dog: The Way of the Samurai (1999)                 | Crime|Drama                         |
| 2793    | Serpico (1973)                                           | Crime|Drama                         |
| 3078    | Making Mr. Right (1987)                                  | Comedy|Romance|Sci-Fi               |
| 3357    | Twins (1988)                                             | Comedy                              |
| 3638    | Lord of the Rings: The Fellowship of the Ring, The (2001)| Adventure|Fantasy                   |
| 3745    | Ice Age (2002)                                           | Adventure|Animation|Children|Comedy |
| 3832    | Star Wars: Episode II - Attack of the Clones (2002)      | Action|Adventure|Sci-Fi|IMAX        |
| 3873    | Minority Report (2002)                                   | Action|Crime|Mystery|Sci-Fi|Thriller|
| 3875    | Sunshine State (2002)                                    | Drama                               |
| 3879    | Pumpkin (2002)                                           | Comedy|Drama|Romance                |
| 3903    | Austin Powers in Goldmember (2002)                       | Comedy                              |
| 3920    | xXx (2002)                                               | Action|Crime|Thriller               |
| 4088    | Return to the Blue Lagoon (1991)                         | Adventure|Romance                   |
| 4089    | Toy Soldiers (1991)                                      | Action|Drama                        |
| 4096    | Die Another Day (2002)                                   | Action|Adventure|Thriller           |
| 4110    | Elling (2001)                                            | Comedy|Drama                        |
| 4111    | I Spit on Your Grave (Day of the Woman) (1978)           | Horror|Thriller                     |
| 4112    | Last Seduction, The (1994)                               | Crime|Drama|Thriller                |
| 4117    | Adaptation (2002)                                        | Comedy|Drama|Romance                |
| 4137    | Lord of the Rings: The Two Towers, The (2002)            | Adventure|Fantasy                   |
| 4141    | Gangs of New York (2002)                                 | Crime|Drama                         |
| 4145    | Body of Evidence (1993)                                  | Drama|Thriller                      |
| 4147    | Duellists, The (1977)                                    | Action|War                          |
| 4158    | Quicksilver (1986)                                       | Drama                               |
| 4167    | King of Comedy, The (1983)                               | Comedy|Drama                        |
| 4192    | Blind Date (1984)                                        | Horror|Thriller                     | 

Here are the results from our recommendation system.
| movieId | title                                                    | genres                              |
|---------|----------------------------------------------------------|-------------------------------------|
| 277     | Shawshank Redemption, The (1994)                         | Crime|Drama                         |
| 463     | Searching for Bobby Fischer (1993)                       | Drama                               |
| 474     | Blade Runner (1982)                                      | Action|Sci-Fi|Thriller              |
| 510     | Silence of the Lambs, The (1991)                         | Crime|Horror|Thriller               |
| 680     | Philadelphia Story, The (1940)                           | Comedy|Drama|Romance                |
| 711     | Notorious (1946)                                         | Film-Noir|Romance|Thriller          |
| 820     | Monty Python's Life of Brian (1979)                      | Comedy                              |
| 856     | Abyss, The (1989)                                        | Action|Adventure|Sci-Fi|Thriller    |
| 863     | Monty Python and the Holy Grail (1975)                   | Adventure|Comedy|Fantasy            |
| 907     | Clockwork Orange, A (1971)                               | Crime|Drama|Sci-Fi|Thriller         |
| 920     | Psycho (1960)                                            | Crime|Horror                        |
| 922     | Godfather: Part II, The (1974)                           | Crime|Drama                         |
| 1243    | Gattaca (1997)                                           | Drama|Sci-Fi|Thriller               |
| 1444    | Labyrinth (1986)                                         | Adventure|Fantasy|Musical           |
| 1529    | Roger & Me (1989)                                        | Documentary                        |
| 2765    | Road Warrior, The (Mad Max 2) (1981)                     | Action|Adventure|Sci-Fi|Thriller    |
| 2982    | Unbreakable (2000)                                       | Drama|Sci-Fi                        |
| 3016    | Traffic (2000)                                           | Crime|Drama|Thriller                |
| 3234    | A.I. Artificial Intelligence (2001)                      | Adventure|Drama|Sci-Fi              |
| 4800    | Lord of the Rings: The Return of the King, The (2003)    | Action|Adventure|Drama|Fantasy      |

Most of the recommended movies are related to the user's viewing history based on their genres. "The Lord of the Rings" is one such recommendation, as they have already watched two movies from this series.
