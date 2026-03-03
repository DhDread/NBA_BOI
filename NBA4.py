# nba_etl_skip_none_tqdm.py
import time
from datetime import datetime, timedelta
import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2, BoxScoreSummaryV3, BoxScoreTraditionalV3
from tqdm import tqdm  # progress bar

# CONFIG
DAYS_BACK = 2
PAUSE_BETWEEN_REQUESTS = 2  # seconds

def fetch_scoreboard(date_str):
    """Fetch scoreboard safely for a given date."""
    try:
        sb = ScoreboardV2(game_date=date_str, league_id='00')
        dfs = sb.get_data_frames()
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            if 'GAME_ID' in df.columns:
                return df
            else:
                print(f"No GAME_ID column found for {date_str}. Columns: {df.columns.tolist()}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching scoreboard {date_str}: {e}")
        return pd.DataFrame()

def fetch_box_score_skip_none(game_id):
    """Fetch box score and skip if any part is None."""
    try:
        summary_dfs = BoxScoreSummaryV3(game_id=game_id).get_data_frames() or []
        traditional_dfs = BoxScoreTraditionalV3(game_id=game_id).get_data_frames() or []

        # If any dataframe is None, skip the game
        if any(df is None for df in summary_dfs + traditional_dfs):
            return None

        return {"summary": summary_dfs, "traditional": traditional_dfs}
    except Exception:
        return None

def main():
    print("========================================")
    print("NBA Full Box Score ETL (Skip None Version with Progress Bar)")
    print("========================================")

    today = datetime.utcnow().date()
    all_scoreboards = []

    # Fetch scoreboards
    for day_idx in range(DAYS_BACK):
        date_obj = today - timedelta(days=day_idx)
        date_str = date_obj.strftime("%m/%d/%Y")
        print(f"Fetching scoreboard for {date_str} (Day {day_idx + 1}/{DAYS_BACK})...")
        df = fetch_scoreboard(date_str)
        if not df.empty:
            all_scoreboards.append(df)
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not all_scoreboards:
        print("No scoreboards found.")
        return

    scoreboards_df = pd.concat(all_scoreboards, ignore_index=True)
    total_games = len(scoreboards_df)
    print(f"Found {total_games} games in {DAYS_BACK} days.")

    all_box_scores = []

    # Fetch box scores with progress bar
    print("Fetching box scores...")
    for idx, row in enumerate(tqdm(scoreboards_df.itertuples(), total=total_games, unit="game")):
        game_id = row.GAME_ID
        data = fetch_box_score_skip_none(game_id)
        if data:
            for dfs in data.values():
                for df in dfs:
                    df['GAME_ID'] = game_id
                    all_box_scores.append(df)
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not all_box_scores:
        print("No box scores fetched.")
        return

    # Combine all and save CSV
    final_df = pd.concat(all_box_scores, ignore_index=True)
    timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
    filename = f"{timestamp}_NBA_Stats.csv"
    final_df.to_csv(filename, index=False)
    print(f"\nSaved all data to CSV: {filename}")

if __name__ == "__main__":
    main()
