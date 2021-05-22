# nfl-predictor

This Flask web-app allows users to input how important they believe certain statistics are in determining the outcome of an NFL game based on the current season's statistics. The app then uses a normalization process to create rankings of the teams; these rankings include Z scores and percentiles that indicate each team's relative standing within the league. The user can then use these rankings to predict head to head matchups and to compare their model's predictions for the upcoming games to that of DraftKings. 

This webapp uses NGINX running as webserver in a docker container.

To run this web app locally, clone the repo, then run `docker compose up --build` and go to http://localhost. Note: Docker must be running on your machine for this to work.

Check out nfl-predictor.ipynb to see how the rankings and predictions are created.

Data is collected from football-reference via the sportsreference api.