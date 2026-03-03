import sqlite3
import pandas as pd
import requests
import os

# ==================================================
# CONFIG
# ==================================================
USE_SQLITE = True
IF_EXISTS = "replace"  # replace | append | fail

SCOREBOARD_URL = "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com/NBA/liveData/scoreboard/todaysScoreboard_00.json"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "nba_full.db")
# ==================================================


def fetch_live_data():
    response = requests.get(SCOREBOARD_URL)
    response.raise_for_status()
    return response.json()


# ==================================================
# BUILD DATASETS
# ==================================================

def build_datasets(data):

    scoreboard = data["scoreboard"]
    games_raw = scoreboard["games"]

    games_rows = []
    teams_rows = []
    periods_rows = []
    leaders_rows = []

    for game in games_raw:
        game_id = game["gameId"]

        # ---------------- GAME TABLE ----------------
        games_rows.append({
            "game_id": game_id,
            "game_date": scoreboard["gameDate"],
            "game_code": game["gameCode"],
            "status": game["gameStatusText"],
            "period": game["period"],
            "game_time_utc": game["gameTimeUTC"]
        })

        # ---------------- TEAM & PERIOD SCORES ----------------
        for side in ["homeTeam", "awayTeam"]:
            team = game[side]
            side_str = "home" if side == "homeTeam" else "away"

            teams_rows.append({
                "game_id": game_id,
                "side": side_str,
                "team_id": team["teamId"],
                "team_name": team["teamName"],
                "team_city": team["teamCity"],
                "team_tricode": team["teamTricode"],
                "wins": team["wins"],
                "losses": team["losses"],
                "score": team["score"]
            })

            for p in team["periods"]:
                periods_rows.append({
                    "game_id": game_id,
                    "side": side_str,
                    "team_id": team["teamId"],
                    "period_number": p["period"],
                    "period_type": p["periodType"],
                    "points": p["score"]
                })

        # ---------------- GAME LEADERS ----------------
        for leader_type in ["homeLeaders", "awayLeaders"]:
            leader = game["gameLeaders"][leader_type]
            side_str = "home" if leader_type == "homeLeaders" else "away"

            leaders_rows.append({
                "game_id": game_id,
                "side": side_str,
                "person_id": leader["personId"],
                "name": leader["name"],
                "jersey": leader["jerseyNum"],
                "position": leader["position"],
                "team_tricode": leader["teamTricode"],
                "points": leader["points"],
                "rebounds": leader["rebounds"],
                "assists": leader["assists"]
            })

    return (
        pd.DataFrame(games_rows),
        pd.DataFrame(teams_rows),
        pd.DataFrame(periods_rows),
        pd.DataFrame(leaders_rows),
    )


# ==================================================
# BUILD READABLE GAME VIEW
# ==================================================

def build_readable_game_view(games_df, teams_df, periods_df):
    # Pivot period scores
    periods_pivot = (
        periods_df
        .pivot_table(
            index=["game_id", "side", "team_id"],
            columns="period_number",
            values="points"
        )
        .reset_index()
    )

    # Rename period columns
    periods_pivot.columns = [
        f"Q{col}" if isinstance(col, int) else col
        for col in periods_pivot.columns
    ]

    # Merge teams + periods
    merged = teams_df.merge(
        periods_pivot,
        on=["game_id", "side", "team_id"],
        how="left"
    )

    # Merge game info
    readable = merged.merge(
        games_df,
        on="game_id",
        how="left"
    )

    return readable.sort_values(["game_id", "side"])


# ==================================================
# SAVE TO SQLITE
# ==================================================

def save_to_sqlite(datasets):
    conn = sqlite3.connect(DB_NAME)
    table_names = ["games", "teams", "period_scores", "game_leaders"]

    for df, table in zip(datasets, table_names):
        df.to_sql(table, conn, if_exists=IF_EXISTS, index=False)
        print(f"Saved table: {table} ({len(df)} rows)")

    conn.commit()
    conn.close()
    print("\nDatabase saved at:", DB_NAME)


# ==================================================
# MAIN
# ==================================================

def main():
    print("Fetching live NBA scoreboard...")
    data = fetch_live_data()

    print("Building datasets...")
    games_df, teams_df, periods_df, leaders_df = build_datasets(data)

    # ---------- READABLE VIEWS ----------
    readable_games = build_readable_game_view(games_df, teams_df, periods_df)
    readable_leaders = leaders_df.sort_values(["game_id", "side"])

    print("\n================ READABLE GAME VIEW ================")
    print(readable_games)

    print("\n================ GAME LEADERS ================")
    print(readable_leaders)

    # ---------- SAVE TO SQLITE ----------
    if USE_SQLITE:
        save_to_sqlite((games_df, teams_df, periods_df, leaders_df))
    else:
        print("\nSQLite saving disabled.")


if __name__ == "__main__":
    main()
