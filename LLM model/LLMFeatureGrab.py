import os
from openai import OpenAI
import pandas as pd

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataset.csv")
df = pd.read_csv(csv_path)

# Function to get LLM confidence
def get_llm_confidence(row):
    prompt = f"""
    Match data:
    Team A winrate recent 5 games: {row['teamA_recent5_winrate']}
    Team B winrate recent 5 games: {row['teamB_recent5_winrate']}
    Team A winrate recent 10 games: {row['teamA_winrate']}
    Team B winrate recent 10 games: {row['teamB_winrate']}
    Team A avg round diff recent 10 games: {row['teamA_avg_round_diff']}
    Team B avg round diff recent 10 games: {row['teamB_avg_round_diff']}
    Team A winstreak recent 10 games: {row['teamA_winstreak']}
    Team B winstreak recent 10 games: {row['teamB_winstreak']}
    
    Based on this, give a confidence score from 0 to 1 for Team A winning.
    Respond with only the score in format "Team A: score", example "Team A: 0.45"
    """
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    text = response.choices[0].message["content"]
    try:
        team_a_conf = float(text.split("Team A:")[1].strip())
    except:
        team_a_conf = 0.5
    
    return team_a_conf

confidence_df = pd.DataFrame({
    "chat_teamA_conf": df.apply(get_llm_confidence, axis=1)
})

print(confidence_df)

confidence_df.to_csv("teamA_conf.csv", index=False)