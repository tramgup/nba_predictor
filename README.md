# nba_predictor
A Python based logistic regression model for predicting NBA games.


## Introduction
The NBA Game Predictor is a project designed to predict the results of NBA games based on historical team performance and statistical data. Using data from the last 30 days, this model provides an answer into which team is more likely to win a particular game.

This project was done entirely using Google Colab, so it is recommended to use a similar environment to run the following program, such as Jupyter notebook.

The data used was retrieved from the SportsData.io API, which is a free API anyone can use by signing up. To use this model you have to sign up for an account and paste into the code your own API Key. Luckily the process is very simple.


## Data
The Data used is from the SportsData.io API, focusing on the most recent 30 days of NBA games. The script is designed to call two distinct endpoints, one for retrieving game data and the other for team statistics. These stats were then averaged to give a more accurate reading.
The 30 day average of the following stats were used for this project:

* Wins
* Points
* Rebounds
* Assists
* Steals
* Blocked Shots
* True shooting percentage


## Predictive Modeling
The python library Scikit-learn was used to train a logistic regression model on the processed data. The model predicts the outcome (the winning team) of NBA games based on the statistical performance of the two teams.

## Prediction
The model predicts game outcomes for any given pair of teams (matchup). Predictions are based on the most recent team performance data available.


## Results
The model demonstrates an accuracy between 67 -73% in predicting game outcomes based on the test data. Furthermore, the model usually selects the strongly favored team based on sportsbetting websites.


## Future Work
Accounting for specific player lineups, as certain players account for the majority of a teams points (Ex: Check whether Lebron James or Steph Curry is playing). There are many other factors that can also be added such as at-home advantage, social media sentiment, player injury status, player trades, as well as factoring in back to back game disadvantages.














