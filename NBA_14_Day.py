import pandas as pd
from datetime import datetime, timedelta, UTC
from nba_api.stats.endpoints import ScoreboardV3
import time
import numpy as np

DAYS_BACK = 56
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

def build_team_dataset(df):
    team_rows = []

    for _, row in df.iterrows():
        # team A
        team_rows.append({
            "team": row["team_a"],
            "date": pd.to_datetime(row["game_date"]),
            "Q1": row["team_a_Q1"],
            "Q2": row["team_a_Q2"],
            "Q3": row["team_a_Q3"],
            "Q4": row["team_a_Q4"],
            "final": row["team_a_score"],
            "opp_final": row["team_b_score"],  # for team A
            "opp_Q1": row["team_b_Q1"],
            "opp_Q2": row["team_b_Q2"],
            "opp_Q3": row["team_b_Q3"],
            "opp_Q4": row["team_b_Q4"],
        })

        # team B
        team_rows.append({
            "team": row["team_b"],
            "date": pd.to_datetime(row["game_date"]),
            "Q1": row["team_b_Q1"],
            "Q2": row["team_b_Q2"],
            "Q3": row["team_b_Q3"],
            "Q4": row["team_b_Q4"],
            "final": row["team_b_score"],
            "opp_final": row["team_a_score"],  # for team B
            "opp_Q1": row["team_a_Q1"],
            "opp_Q2": row["team_a_Q2"],
            "opp_Q3": row["team_a_Q3"],
            "opp_Q4": row["team_a_Q4"],
        })

    team_df = pd.DataFrame(team_rows)
    team_df = team_df.sort_values(["team", "date"])

    return team_df
    
def compute_stats(team_df):

    results = []

    for team, group in team_df.groupby("team"):

        group = group.sort_values("date")

        stats = {"team": team}
        
        stats["def_avg"] = group["opp_final"].mean()
        #stats["Q1_def_avg"] = group["Q1"].mean()  # placeholder (we'll fix below)
        
        stats["Q1_def_avg"] = group["opp_Q1"].mean()
        stats["Q2_def_avg"] = group["opp_Q2"].mean()
        stats["Q3_def_avg"] = group["opp_Q3"].mean()
        stats["Q4_def_avg"] = group["opp_Q4"].mean()

        for col in ["Q1", "Q2", "Q3", "Q4", "final"]:

            values = group[col]

            # basic stats
            stats[f"{col}_low"] = values.min()
            stats[f"{col}_avg"] = values.mean()
            stats[f"{col}_high"] = values.max()
            stats[f"{col}_std"] = values.std()
            

            # 🔥 average point distance between consecutive games
            diffs = values.diff().abs().dropna()

            stats[f"{col}_avg_diff"] = diffs.mean() if len(diffs) > 0 else 0

        results.append(stats)

    return pd.DataFrame(results)

def add_advanced_metrics(team_df, team_stats):

    advanced_rows = []

    for team, group in team_df.groupby("team"):

        group = group.sort_values("date")

        # Last 3 games average (final score)
        last3 = group["final"].tail(3)
        last3_avg = last3.mean() if len(last3) > 0 else 0

        # Q1 win rate
        q1_wins = 0
        total_games = len(group)

        for i in range(len(group)):
            row = group.iloc[i]

            # need original df comparison (so rebuild opponent lookup)
            # easiest: compare Q1 vs opponent using original df
            # we'll approximate using diff logic later

        # Better way: rebuild using original df
        q1_results = []

        for _, row in group.iterrows():
            q1_results.append(row["Q1"])

        # We'll compute win rate later properly in prediction section

        advanced_rows.append({
            "team": team,
            "last3_avg": last3_avg
        })

    advanced_df = pd.DataFrame(advanced_rows)

    return team_stats.merge(advanced_df, on="team")

def compute_q1_win_rate(df):

    records = []

    for _, row in df.iterrows():

        # team A
        records.append({
            "team": row["team_a"],
            "win": 1 if row["team_a_Q1"] > row["team_b_Q1"] else 0
        })

        # team B
        records.append({
            "team": row["team_b"],
            "win": 1 if row["team_b_Q1"] > row["team_a_Q1"] else 0
        })

    temp = pd.DataFrame(records)

    q1_win_rate = temp.groupby("team")["win"].mean().reset_index()
    q1_win_rate.rename(columns={"win": "Q1_win_rate"}, inplace=True)

    return q1_win_rate
    
def compute_h2h_win_rate(df):

    records = []

    for _, row in df.iterrows():

        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]

        # team A record
        records.append({
            "team": team_a,
            "opponent": team_b,
            "win": 1 if winner == team_a else 0,
            "game": 1
        })

        # team B record
        records.append({
            "team": team_b,
            "opponent": team_a,
            "win": 1 if winner == team_b else 0,
            "game": 1
        })

    h2h_df = pd.DataFrame(records)

    # aggregate
    h2h_stats = (
        h2h_df
        .groupby(["team", "opponent"])
        .agg(
            wins=("win", "sum"),
            games=("game", "sum")
        )
        .reset_index()
    )

    # win rate
    h2h_stats["h2h_win_rate"] = h2h_stats["wins"] / h2h_stats["games"]

    # 🔥 formatted column (what you want)
    h2h_stats["h2h_record"] = (
        h2h_stats["wins"].astype(int).astype(str)
        + "/"
        + h2h_stats["games"].astype(int).astype(str)
    )

    return h2h_stats
    
def get_h2h_rate(h2h_df, team, opponent):

    row = h2h_df[
        (h2h_df["team"] == team) &
        (h2h_df["opponent"] == opponent)
    ]

    if not row.empty:
        return row.iloc[0]["h2h_win_rate"]
    
    return 0.5  # default if no history

def build_predictions(df, team_stats, h2h_df):
    predictions = []

    for _, row in df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]

        # skip if stats missing
        if team_a not in team_stats["team"].values or team_b not in team_stats["team"].values:
            continue

        stats_a = team_stats[team_stats["team"] == team_a].iloc[0]
        stats_b = team_stats[team_stats["team"] == team_b].iloc[0]
        
        h2h_a = get_h2h_rate(h2h_df, team_a, team_b)
        h2h_row = h2h_df[
            (h2h_df["team"] == team_a) &
            (h2h_df["opponent"] == team_b)
        ]

        h2h_record = h2h_row.iloc[0]["h2h_record"] if not h2h_row.empty else "0/0"
        h2h_b = 1 - h2h_a

        # function to predict points per quarter
        def predict_q(team_stats, opp_stats, q):
            return team_stats[f"{q}_avg"] * 0.6 + opp_stats[f"{q}_def_avg"] * 0.4

        # quarter predictions
        a_q1 = predict_q(stats_a, stats_b, "Q1")
        a_q2 = predict_q(stats_a, stats_b, "Q2")
        a_q3 = predict_q(stats_a, stats_b, "Q3")
        a_q4 = predict_q(stats_a, stats_b, "Q4")

        b_q1 = predict_q(stats_b, stats_a, "Q1")
        b_q2 = predict_q(stats_b, stats_a, "Q2")
        b_q3 = predict_q(stats_b, stats_a, "Q3")
        b_q4 = predict_q(stats_b, stats_a, "Q4")

        # total predicted scores
        pred_a = a_q1 + a_q2 + a_q3 + a_q4
        pred_b = b_q1 + b_q2 + b_q3 + b_q4

        predicted_winner = team_a if pred_a > pred_b else team_b

        # --- 🔥 WIN PROBABILITY ---
        margin = pred_a - pred_b
        base_prob_a = 1 / (1 + np.exp(-margin / 10))

        # blend model + head-to-head
        prob_a = (base_prob_a * 0.7) + (h2h_a * 0.3)
        prob_b = 1 - prob_a

        # --- 🔥 CONFIDENCE ---
        max_prob = max(prob_a, prob_b)
        if max_prob >= 0.7:
            confidence = "High"
        elif max_prob >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        # --- RESULT CHECKS ---
        winner_match = "Yes" if predicted_winner == row["winner"] else "No"

        diff_a = abs(pred_a - row["team_a_score"])
        diff_b = abs(pred_b - row["team_b_score"])
        avg_diff = (diff_a + diff_b) / 2

        if avg_diff <= 5:
            score_match = "Excellent"
        elif avg_diff <= 10:
            score_match = "Good"
        elif avg_diff <= 15:
            score_match = "Decent"
        else:
            score_match = "Poor"

        predictions.append({
            "game_date": row["game_date"],
            "team_a": team_a,
            "team_b": team_b,

            # quarter predictions
            "team_a_Q1_pred": round(a_q1),
            "team_a_Q2_pred": round(a_q2),
            "team_a_Q3_pred": round(a_q3),
            "team_a_Q4_pred": round(a_q4),

            "team_b_Q1_pred": round(b_q1),
            "team_b_Q2_pred": round(b_q2),
            "team_b_Q3_pred": round(b_q3),
            "team_b_Q4_pred": round(b_q4),

            # totals
            "pred_team_a_score": round(pred_a),
            "pred_team_b_score": round(pred_b),

            # actual
            "actual_team_a_score": row["team_a_score"],
            "actual_team_b_score": row["team_b_score"],
            "actual_winner": row["winner"],

            # predictions
            "predicted_winner": predicted_winner,

            # 🔥 NEW PROBABILITY FIELDS
            "team_a_win_prob": round(prob_a * 100, 1),
            "team_b_win_prob": round(prob_b * 100, 1),
            "confidence": confidence,

            # 🔥 comparison metrics
            "Winning_Team_Match": winner_match,
            "Score_Match": score_match,
            "Score_Diff_A": round(diff_a, 1),
            "Score_Diff_B": round(diff_b, 1),
            "Avg_Score_Diff": round(avg_diff, 1),
            "h2h_record": h2h_record
        })

    return pd.DataFrame(predictions)

def fetch_today_games():

    today = datetime.now(UTC).date()
    date_str = today.strftime("%Y-%m-%d")

    try:
        sb = ScoreboardV3(game_date=date_str, timeout=10)
        data = sb.get_dict()
        games = data.get("scoreboard", {}).get("games", [])

        rows = []

        for g in games:

            home = g["homeTeam"]
            away = g["awayTeam"]

            rows.append({
                "game_date": date_str,
                "team_a": home["teamName"],
                "team_b": away["teamName"]
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Error fetching today's games: {e}")
        return pd.DataFrame()

def predict_today(today_df, team_stats, h2h_df):

    preds = []

    for _, row in today_df.iterrows():

        team_a = row["team_a"]
        team_b = row["team_b"]

        # skip if missing stats (safety)
        if team_a not in team_stats["team"].values or team_b not in team_stats["team"].values:
            continue

        stats_a = team_stats[team_stats["team"] == team_a].iloc[0]
        stats_b = team_stats[team_stats["team"] == team_b].iloc[0]
        
        h2h_a = get_h2h_rate(h2h_df, team_a, team_b)
        h2h_row = h2h_df[
            (h2h_df["team"] == team_a) &
            (h2h_df["opponent"] == team_b)
        ]

        h2h_record = h2h_row.iloc[0]["h2h_record"] if not h2h_row.empty else "0/0"
        h2h_b = 1 - h2h_a
        
        # --- QUARTER PREDICTIONS ---
        def predict_q(team_stats, opp_stats, q):
            return (
                team_stats[f"{q}_avg"] * 0.6 +
                opp_stats[f"{q}_def_avg"] * 0.4
            )

        a_q1 = predict_q(stats_a, stats_b, "Q1")
        a_q2 = predict_q(stats_a, stats_b, "Q2")
        a_q3 = predict_q(stats_a, stats_b, "Q3")
        a_q4 = predict_q(stats_a, stats_b, "Q4")

        b_q1 = predict_q(stats_b, stats_a, "Q1")
        b_q2 = predict_q(stats_b, stats_a, "Q2")
        b_q3 = predict_q(stats_b, stats_a, "Q3")
        b_q4 = predict_q(stats_b, stats_a, "Q4")

        # --- FINAL SCORE ---
        pred_a = a_q1 + a_q2 + a_q3 + a_q4
        pred_b = b_q1 + b_q2 + b_q3 + b_q4

        predicted_winner = team_a if pred_a > pred_b else team_b

        # --- 🔥 WIN PROBABILITY ---
        margin = pred_a - pred_b

        base_prob_a = 1 / (1 + np.exp(-margin / 10))

        # blend model + head-to-head
        prob_a = (base_prob_a * 0.7) + (h2h_a * 0.3)
        prob_b = 1 - prob_a

        # --- 🔥 CONFIDENCE LEVEL ---
        max_prob = max(prob_a, prob_b)

        if max_prob >= 0.7:
            confidence = "High"
        elif max_prob >= 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        preds.append({
            "game_date": row["game_date"],
            "team_a": team_a,
            "team_b": team_b,

            # quarter predictions
            "team_a_Q1_pred": round(a_q1),
            "team_a_Q2_pred": round(a_q2),
            "team_a_Q3_pred": round(a_q3),
            "team_a_Q4_pred": round(a_q4),

            "team_b_Q1_pred": round(b_q1),
            "team_b_Q2_pred": round(b_q2),
            "team_b_Q3_pred": round(b_q3),
            "team_b_Q4_pred": round(b_q4),

            # final prediction
            "pred_team_a_score": round(pred_a),
            "pred_team_b_score": round(pred_b),

            "predicted_winner": predicted_winner,

            # 🔥 NEW FIELDS
            "team_a_win_prob": round(prob_a * 100, 1),
            "team_b_win_prob": round(prob_b * 100, 1),
            "confidence": confidence,
            "h2h_record": h2h_record
        })

    return pd.DataFrame(preds) 
    
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
    
    team_df = build_team_dataset(df)

    team_stats = compute_stats(team_df)
    
    # add advanced metrics
    team_stats = add_advanced_metrics(team_df, team_stats)

    # add Q1 win rate
    q1_rates = compute_q1_win_rate(df)
    team_stats = team_stats.merge(q1_rates, on="team")

    # build predictions
    # 🔥 Step 5 - compute H2H FIRST
    h2h_df = compute_h2h_win_rate(df)

    # 🔥 Step 6 - pass it into predictions
    predictions = build_predictions(df, team_stats, h2h_df)
    
    # After predictions = build_predictions(df, team_stats)

    # 🔥 Summary statistics
    total_games = len(predictions)

    # Count "Yes" for Winning_Team_Match
    winning_match_count = (predictions["Winning_Team_Match"] == "Yes").sum()
    winning_match_rate = (winning_match_count / total_games) * 100 if total_games > 0 else 0

    # Count "Excellent" for Score_Match
    excellent_count = (predictions["Score_Match"] == "Excellent").sum()
    excellent_rate = (excellent_count / total_games) * 100 if total_games > 0 else 0

    summary_stats = {
    "Total_Games": total_games,
    "Winning_Team_Match_Count": winning_match_count,
    "%_Correct_Rate": round(winning_match_rate, 1),
    "Score_Excellent_Count": excellent_count,
    "%_Excellent_Rate": round(excellent_rate, 1)
    }

    
    today_games = fetch_today_games()
    today_predictions = predict_today(today_games, team_stats, h2h_df)

    print("\nGames Sample:")
    print(df.head())
    print("\nTeam analysis sample:")
    print(team_stats.head())

#-- EXCEL Version
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"NBA_last_{DAYS_BACK}_days_{timestamp}.xlsx"

    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        
        # Sheet 1 - Today's Predictions
        today_predictions.to_excel(writer, sheet_name="Today's Predictions", index=False)
        
        # Sheet 2 - Historical Predictions
        predictions.to_excel(writer, sheet_name="Historical Predictions", index=False)
        
        # Sheet 3 - Team Analysis
        team_stats.to_excel(writer, sheet_name="Team Analysis", index=False)
        
        # Sheet 4 - Raw Games
        df.to_excel(writer, sheet_name="Games", index=False)

        # Optional Sheet 5 - Summary Stats
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Sheet 6 - Head-to-Head Stats
        h2h_matrix = h2h_df.pivot(
            index="team",
            columns="opponent",
            values="h2h_record"
        )
        h2h_pivot = h2h_df.pivot(index="team", columns="opponent", values="h2h_win_rate")
        h2h_df.to_excel(writer, sheet_name="H2H Stats", index=False)
        h2h_matrix.to_excel(writer, sheet_name="H2H Matrix")

# -- CSV Method
#    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#    filename = f"NBA_last_{DAYS_BACK}_days_{timestamp}.csv"
#
#    with open(filename, "w", newline="") as f:
#
#        df.to_csv(f, index=False)
#
#        f.write("\n\n\n===== TEAM ANALYSIS =====\n")
#        team_stats.to_csv(f, index=False)
#
#        f.write("\n\n\n===== HISTORICAL PREDICTIONS =====\n")
#        predictions.to_csv(f, index=False)
#        
#        f.write("\n\n===== HISTORICAL PREDICTIONS SUMMARY =====\n")
#        pd.DataFrame([summary_stats]).to_csv(f, index=False)
#
#        f.write("\n\n\n===== TODAY'S PREDICTIONS =====\n")
#        today_predictions.to_csv(f, index=False)

    print(f"\nSaved combined data to {filename}")


if __name__ == "__main__":
    main()
