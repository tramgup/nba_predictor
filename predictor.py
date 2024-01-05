import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# loading dataset
nba_data = pd.read_csv('nba_games_with_team_stats_last_30_days.csv')

# calculating win rate
def calculate_win_rate(team, df):
    games_played = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
    wins = games_played[((games_played['HomeTeam'] == team) & (games_played['HomeTeamScore'] > games_played['AwayTeamScore'])) |
                        ((games_played['AwayTeam'] == team) & (games_played['AwayTeamScore'] > games_played['HomeTeamScore']))]
    return len(wins) / len(games_played) if len(games_played) > 0 else 0

# calculating average score
def calculate_average_score(team, df):
    home_games = df[df['HomeTeam'] == team]
    away_games = df[df['AwayTeam'] == team]
    total_points = home_games['HomeTeamScore'].sum() + away_games['AwayTeamScore'].sum()
    total_games = len(home_games) + len(away_games)
    return total_points / total_games if total_games > 0 else 0

# calculating the average of a given stat
def calculate_stat_average(df, team, stat):
    home_stats = df[df['HomeTeam'] == team][stat]
    away_stats = df[df['AwayTeam'] == team][stat]
    total_stat = home_stats.sum() + away_stats.sum()
    total_games = len(home_stats) + len(away_stats)
    return total_stat / total_games if total_games > 0 else 0

# List of stats in the csv file that are to be used as
# can add more stats if needed based on the csv file data
stats = [
    'Rebounds', 'Assists', 'Steals', 'BlockedShots', 'TrueShootingPercentage'
]

# calculating the stats and storing into a dict
teams = nba_data['HomeTeam'].unique()
team_stats = {
    team: {
        'win_rate': calculate_win_rate(team, nba_data),
        'avg_score': calculate_average_score(team, nba_data),
        **{stat: calculate_stat_average(nba_data, team, stat) for stat in stats}
    }
    for team in teams
}

# Convert the dict to pandas DataFrame
team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index')

# preping the dataset with features for model training
# training based on win rate, avg score, and avg of other game stats
model_data_list = []
for index, row in nba_data.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    home_team_win = int(row['HomeTeamScore'] > row['AwayTeamScore'])
    row_data = {
        'HomeTeamWin': home_team_win,
        'HomeTeamWinRate': team_stats_df.loc[home_team, 'win_rate'],
        'AwayTeamWinRate': team_stats_df.loc[away_team, 'win_rate'],
        'HomeTeamAvgScore': team_stats_df.loc[home_team, 'avg_score'],
        'AwayTeamAvgScore': team_stats_df.loc[away_team, 'avg_score'],
        **{f'Home{stat}': team_stats_df.loc[home_team, stat] for stat in stats},
        **{f'Away{stat}': team_stats_df.loc[away_team, stat] for stat in stats}
    }
    model_data_list.append(row_data)

model_data = pd.concat([pd.DataFrame([row]) for row in model_data_list], ignore_index=True)

# define features and target for the model
X = model_data.drop('HomeTeamWin', axis=1)
y = model_data['HomeTeamWin']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# spliting the scaled dataset and training the model
# 20% of data used for testing
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

# printing accuracy of the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# function to predict the match outcome
def predict_match_outcome(home_team, away_team, model, team_stats_df, scaler):
    match_features = {
        'HomeTeamWinRate': team_stats_df.loc[home_team, 'win_rate'], # Team 1
        'AwayTeamWinRate': team_stats_df.loc[away_team, 'win_rate'], # Team 2
        'HomeTeamAvgScore': team_stats_df.loc[home_team, 'avg_score'],
        'AwayTeamAvgScore': team_stats_df.loc[away_team, 'avg_score'],
        **{f'Home{stat}': team_stats_df.loc[home_team, stat] for stat in stats},
        **{f'Away{stat}': team_stats_df.loc[away_team, stat] for stat in stats}
    }
    features_df = pd.DataFrame([match_features])

    # matching colum with training data
    features_df = features_df[X.columns]

    features_scaled = scaler.transform(features_df)  # Scale features before prediction
    prediction = model.predict(features_scaled)
    return f"Predicted Winner: {home_team}" if prediction[0] == 1 else f"Predicted Winner: {away_team}"

# Predict the outcome of a match between two teams using NBA team abbreviations. The team abbreviations are provided in another file in the repo
# Replace "Team 1" and "Team 2" with the two teams you want to check. EX: DEN, MIA, GS ...
predicted_winner = predict_match_outcome('Team 1', 'Team 2', model, team_stats_df, scaler)
print(predicted_winner)
