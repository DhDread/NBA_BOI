import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import ScoreboardV2
import time

# =============================
# CONFIG
# =============================
DAYS_BACK = 14
PAUSE_BETWEEN_REQUESTS = 1  # seconds to avoid rate limiting

# =============================
# FUNCTIONS
# =============================
def fetch_scoreboard(date_obj):
    """Fetch scoreboard for a specific date."""
    date_str = date_obj.strftime("%m/%d/%Y")  # NBA_API expects MM/DD/YYYY
    try:
        sb = ScoreboardV2(game_date=date_str, league_id='00')
        data = sb.get_dict()
        games = data.get('resultSets', [])[0].get('rowSet', [])
        headers = data.get('resultSets', [])[0].get('headers', [])
        df = pd.DataFrame(games, columns=headers)
        return df
    except Exception as e:
        print(f"Error fetching {date_str}: {e}")
        return pd.DataFrame()

def build_game_rows(scoreboard_df, date_obj):
    """Transform ScoreboardV2 dataframe into rows with the fields we want."""
    rows = []
    if scoreboard_df.empty:
        return rows

    for _, game in scoreboard_df.iterrows():
        # Teams and scores
        home_team = game['HOME_TEAM_NAME']
        away_team = game['VISITOR_TEAM_NAME']
        home_score = int(game['HOME_TEAM_SCORE'])
        away_score = int(game['VISITOR_TEAM_SCORE'])
        winner = home_team if home_score > away_score else away_team

        # Period scores
        periods = []
        for i in range(1, 7):  # Q1–Q6
            home_col = f'PTS_Q{i}_HOME'
            away_col = f'PTS_Q{i}_VISITOR'
            # Some columns may not exist, default to 0
            home_period = int(game.get(home_col, 0)) if home_col in game else 0
            away_period = int(game.get(away_col, 0)) if away_col in game else 0
            periods.append((home_period, away_period))

        row = {
            'game_date': date_obj.strftime("%Y-%m-%d"),
            'team_a': home_team,
            'team_b': away_team,
            'team_a_score': home_score,
            'team_b_score': away_score,
            'winner': winner,
            'team_a_Q1': periods[0][0],
            'team_a_Q2': periods[1][0],
            'team_a_Q3': periods[2][0],
            'team_a_Q4': periods[3][0],
            'team_a_Q5': periods[4][0],
            'team_a_Q6': periods[5][0],
            'team_b_Q1': periods[0][1],
            'team_b_Q2': periods[1][1],
            'team_b_Q3': periods[2][1],
            'team_b_Q4': periods[3][1],
            'team_b_Q5': periods[4][1],
            'team_b_Q6': periods[5][1],
        }
        rows.append(row)
    return rows

# =============================
# MAIN
# =============================
def main():
    print("=====================================")
    print(f"Fetching last {DAYS_BACK} days of NBA games...")
    print("=====================================")

    today = datetime.utcnow().date()
    all_rows = []

    for i in range(DAYS_BACK):
        date_obj = today - timedelta(days=i)
        print(f"Fetching scoreboard for {date_obj}...")
        sb_df = fetch_scoreboard(date_obj)
        game_rows = build_game_rows(sb_df, date_obj)
        all_rows.extend(game_rows)
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not all_rows:
        print("No games found in the last 14 days.")
        return

    df = pd.DataFrame(all_rows)
    print("\n==== Sample Output ====")
    print(df.head())

    # Save to CSV
    filename = f"NBA_last_{DAYS_BACK}_days.csv"
    df.to_csv(filename, index=False)
    print(f"\nSaved all games to {filename}")

if __name__ == "__main__":
    main()