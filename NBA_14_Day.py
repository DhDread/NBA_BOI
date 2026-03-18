import pandas as pd
from datetime import datetime, timedelta, UTC
from nba_api.stats.endpoints import ScoreboardV3
import time

DAYS_BACK = 14
PAUSE_BETWEEN_REQUESTS = 2.5


def fetch_scoreboard(date_obj):

    date_str = date_obj.strftime("%Y-%m-%d")

    try:
        sb = ScoreboardV3(game_date=date_str, timeout=10)

        data = sb.get_dict()

        games = data.get("scoreboard", {}).get("games", [])

        if not games:
            return []

        rows = []

        for g in games:

            home = g["homeTeam"]
            away = g["awayTeam"]

            home_name = home["teamName"]
            away_name = away["teamName"]

            home_score = int(home.get("score", 0))
            away_score = int(away.get("score", 0))
            
            if g.get("gameStatusText") != "Final":
                continue

            winner = home_name if home_score > away_score else away_name

            row = {
                "game_date": date_str,

                "team_a": home_name,
                "team_b": away_name,

                "team_a_score": home_score,
                "team_b_score": away_score,

                "winner": winner,

                "team_a_Q1": home.get("periods", [{}])[0].get("score", 0),
                "team_a_Q2": home.get("periods", [{}, {}])[1].get("score", 0),
                "team_a_Q3": home.get("periods", [{}, {}, {}])[2].get("score", 0),
                "team_a_Q4": home.get("periods", [{}, {}, {}, {}])[3].get("score", 0),

                "team_b_Q1": away.get("periods", [{}])[0].get("score", 0),
                "team_b_Q2": away.get("periods", [{}, {}])[1].get("score", 0),
                "team_b_Q3": away.get("periods", [{}, {}, {}])[2].get("score", 0),
                "team_b_Q4": away.get("periods", [{}, {}, {}, {}])[3].get("score", 0),
            }

            rows.append(row)

        return rows

    except Exception as e:
        print(f"Error fetching {date_str}: {e}")
        return []


def main():

    print("=====================================")
    print(f"Fetching last {DAYS_BACK} days of NBA games...")
    print("=====================================")

    today = datetime.now(UTC).date()

    all_rows = []

    for i in range(DAYS_BACK):

        date_obj = today - timedelta(days=i)

        print(f"Fetching {date_obj}...")

        rows = fetch_scoreboard(date_obj)

        all_rows.extend(rows)

        time.sleep(PAUSE_BETWEEN_REQUESTS)

    if not all_rows:
        print("No games found.")
        return

    df = pd.DataFrame(all_rows)

    print("\nSample:")
    print(df.head())

    filename = f"NBA_last_{DAYS_BACK}_days.csv"
    df.to_csv(filename, index=False)

    print(f"\nSaved to {filename}")


if __name__ == "__main__":
    main()
