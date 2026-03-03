import time
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from nba_api.stats.endpoints import ScoreboardV2, BoxScoreSummaryV3, BoxScoreTraditionalV3
from requests.exceptions import RequestException

# ------------------- CONFIG -------------------
DB_PATH = "nba_full.db"
PAUSE_BETWEEN_REQUESTS = 2  # seconds between API calls
MAX_RETRIES = 3
RETRY_PAUSE = 5  # seconds
DAYS_BACK = 14

# ------------------- FUNCTIONS -------------------
def fetch_scoreboard(date_str):
    """Fetch scoreboard for a given date using ScoreboardV2."""
    print(f"Fetching scoreboard {date_str}...")
    for attempt in range(MAX_RETRIES):
        try:
            sb = ScoreboardV2(game_date=date_str)
            df_list = sb.get_data_frames()
            for df in df_list:
                if 'GAME_ID' in df.columns:
                    return df
            print(f"No valid GAME_ID found for {date_str}")
            return pd.DataFrame()
        except RequestException as e:
            print(f"Retry {attempt+1} for {date_str} after error: {e}")
            time.sleep(RETRY_PAUSE)
    return pd.DataFrame()

def fetch_box_score(game_id):
    """Fetch box score safely. Returns dict of DataFrames, or None if unavailable."""
    print(f"Fetching box score for {game_id}...")
    for attempt in range(MAX_RETRIES):
        try:
            summary = BoxScoreSummaryV3(game_id=game_id).get_data_frames()
            # Traditional V3 may fail if data is incomplete
            try:
                traditional = BoxScoreTraditionalV3(game_id=game_id).get_data_frames()
            except Exception:
                print(f"Warning: Traditional box score missing for {game_id}")
                traditional = []
            return {"summary": summary, "traditional": traditional}
        except RequestException as e:
            print(f"Retry {attempt+1} for {game_id} after error: {e}")
            time.sleep(RETRY_PAUSE)
    print(f"Failed to fetch box score for {game_id}")
    return None

def save_to_db(df_dict, db_path=DB_PATH):
    """Save all DataFrames to SQLite database."""
    conn = sqlite3.connect(db_path)
    for key, dfs in df_dict.items():
        for i, df in enumerate(dfs):
            table_name = f"{key}_{i}" if len(dfs) > 1 else key
            df.to_sql(table_name, conn, if_exists="append", index=False)
    conn.close()

# ------------------- MAIN -------------------
def main():
    print("========================================")
    print("NBA Full Box Score ETL (V3)")
    print("========================================")

    today = datetime.utcnow().date()
    all_scoreboards = []

    # Step 1: Fetch scoreboards
    for n in range(DAYS_BACK):
        date_obj = today - timedelta(days=n)
        date_str = date_obj.strftime("%m/%d/%Y")
        df = fetch_scoreboard(date_str)
        if not df.empty:
            all_scoreboards.append(df)
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not all_scoreboards:
        print("No scoreboards found.")
        return

    scoreboards_df = pd.concat(all_scoreboards, ignore_index=True)
    print(f"Found {len(scoreboards_df)} games in {DAYS_BACK} days.")

    # Step 2: Fetch box scores
    box_scores_data = {}
    for idx, row in scoreboards_df.iterrows():
        game_id = row['GAME_ID']
        data = fetch_box_score(game_id)
        if data:
            box_scores_data[game_id] = data
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    # Step 3: Flatten and save
    flattened_data = {}
    for game_id, data_dict in box_scores_data.items():
        for key, dfs in data_dict.items():
            table_key = f"{key}_{game_id}"
            flattened_data[table_key] = dfs

    save_to_db(flattened_data)
    print(f"Database saved at: {DB_PATH}")

if __name__ == "__main__":
    main()
