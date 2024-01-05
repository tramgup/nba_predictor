import requests
import pandas as pd
from datetime import datetime, timedelta

#create and use your own free API Key from sportsdataIO
API_KEY = 'insert key here'

# fetching game data
def fetch_data(date):
    url = f"https://api.sportsdata.io/v3/nba/scores/json/GamesByDate/{date}"
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching game data for {date}: {response.status_code}")
        return None

# fetching team game stats
def fetch_team_game_stats(date):
    url = f"https://api.sportsdata.io/v3/nba/scores/json/TeamGameStatsByDate/{date}?key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching team game stats for {date}: {response.status_code}")
        return None

# retrieving game data for last 30 days.
# can increase interval if needed for more accurate readings
def last_30_days_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    all_game_data = []
    all_team_stats_data = []

    for single_date in (start_date + timedelta(n) for n in range(30)):
        date_str = single_date.strftime("%Y-%m-%d")
        game_data = fetch_data(date_str)
        team_stats_data = fetch_team_game_stats(date_str)

        if game_data:
            all_game_data.extend(game_data)
        if team_stats_data:
            all_team_stats_data.extend(team_stats_data)

    return all_game_data, all_team_stats_data

# getting game data for past 30 days
game_data, team_stats_data = last_30_days_data()

# convert to pandas DataFrames
df_game_data = pd.DataFrame(game_data)
df_team_stats_data = pd.DataFrame(team_stats_data)

# merging both data sets based on the game ID
merged_df = pd.merge(df_game_data, df_team_stats_data, how='left', on='GameID')

# saving to a csv file to access later
merged_df.to_csv('nba_games_with_team_stats_last_30_days.csv', index=False)
