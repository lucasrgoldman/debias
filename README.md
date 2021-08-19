# nfl-predictor

This Flask web-app allows users to input how important they believe certain statistics are in determining the outcome of an NFL game based on the current season's statistics. The app then uses a normalization process to create rankings of the teams; these rankings include Z scores and percentiles that indicate each team's relative standing within the league. The user can then use these rankings to predict head to head matchups and to compare their model's predictions for the upcoming games to that of DraftKings. 

This webapp uses NGINX running as webserver in a docker container.

To run this web app locally, clone the repo, then run `docker compose up --build` and go to http://localhost. Note: Docker must be running on your machine for this to work.

Check out nfl-predictor.ipynb to see how the rankings and predictions are created.

Data is collected from football-reference via the sportsreference api.

## Screenshots

### Power Rank Teams

<img width="1609" alt="Screen Shot 2021-08-19 at 11 56 30 AM" src="https://user-images.githubusercontent.com/58446351/130102274-d60aed7e-1d1f-46af-b647-eb87e39df24c.png">

### Predict Matchups and Compare Predictions to Vegas

<img width="1650" alt="Screen Shot 2021-08-19 at 11 56 47 AM" src="https://user-images.githubusercontent.com/58446351/130102231-3605763c-1cd0-4c0d-bb79-b5b30d6b089c.png">

