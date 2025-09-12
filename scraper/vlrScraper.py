import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime

BASE_URL = "https://www.vlr.gg/matches"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Scrape Matches Function
# ------------------------------------------------------------------------------------------------------------------------

def scrape_matches(max_pages=10):
    tier1_matches = []

    for page in range(1, max_pages+1):
        url = f"{BASE_URL}/?page={page}"
        print(f"Scraping {url}...")
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        matches = soup.select("a.match-item")
        if not matches:
            print(f"No matches on page {page}, stopping.")
            break  # stop if empty page (no more matches)

        for m in matches:
            event = m.select_one(".match-item-event")
            teams = m.select(".match-item-vs-team-name")

            if event:
                event_text = event.get_text(strip=True)

                cond1 = ("Masters" in event_text)
                cond2 = ("Champions" in event_text)

                if cond1 or cond2:
                    team_names = [t.get_text(strip=True) for t in teams]
                    
                    if any("TBD" in name for name in team_names):
                        continue
                    
                    tier1_matches.append({
                        "event": event_text,
                        "teams": " vs ".join(team_names),
                        "link": "https://www.vlr.gg" + m["href"]
                    })

    return tier1_matches


# Test Scraper
matches = scrape_matches()  
print("Found", len(matches), "Tier 1 matches:\n")
for match in matches:
    print(match["event"], "-", match["teams"], "-", match["link"])


# Scrape Teams From Matches
# ------------------------------------------------------------------------------------------------------------------------

def get_teams_from_match(match_url):
    response = requests.get(match_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    teams = []
    for t in soup.select("a.match-header-link"):
        name = t.select_one(".wf-title-med").get_text(strip=True)
        url = "https://www.vlr.gg" + t["href"]
        teams.append({"name": name, "url": url})

    return teams

# Test Scraper
teams = get_teams_from_match(matches[0]["link"])
print(teams)



# Scrape Individual Teams
# ------------------------------------------------------------------------------------------------------------------------

def get_completed_matches_url(team_url):
    # team_url looks like: https://www.vlr.gg/team/624/paper-rex
    return team_url.replace("/team/", "/team/matches/") + "/?group=completed"

def get_team_history(team_url, max_matches=50):
    response = requests.get(team_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    history = []
    rows = soup.select("a.wf-card.fc-flex.m-item")[:max_matches]

    for row in rows:
        # Event
        event_div = row.select_one(".m-item-event")
        event_text = " ".join(event_div.get_text(strip=True).split()) if event_div else None

        # Teams
        team1 = row.select_one(".m-item-team .m-item-team-name")
        team2 = row.select_one(".m-item-team.mod-right .m-item-team-name")
        team1_name = team1.get_text(strip=True) if team1 else None
        team2_name = team2.get_text(strip=True) if team2 else None

        # Score & result
        score_div = row.select_one(".m-item-result")
        score = None
        result = None
        if score_div:
            spans = score_div.select("span")
            if len(spans) >= 2:
                score = f"{spans[0].get_text(strip=True)}-{spans[1].get_text(strip=True)}"
            if "mod-win" in score_div.get("class", []):
                result = "Win"
            elif "mod-loss" in score_div.get("class", []):
                result = "Loss"

        # Date & time
        date_div = row.select_one(".m-item-date div")
        match_dt = None
        if date_div:
            date_text = date_div.get_text(strip=True)
            try:
                match_dt = datetime.strptime(date_text, "%Y/%m/%d")  # only parse the date
            except Exception as e:
                print("Error parsing datetime:", e)



        history.append({
            "event": event_text,
            "team1": team1_name,
            "team2": team2_name,
            "score": score,
            "result": result,
            "datetime": match_dt
        })

    return history


team_data = get_team_history(get_completed_matches_url(teams[0]["url"]))
print(team_data)

def compute_team_features(team, last_n=10):
    # Only use last N matches
    recent = team[:last_n]

    wins, losses, round_diff_total = 0, 0, 0

    for match in recent:
        if match["result"] == "Win":
            wins += 1
        elif match["result"] == "Loss":
            losses += 1

        # Parse "3-1" → (3,1)
        if match["score"]:
            try:
                left, right = match["score"].split("-")
                left, right = int(left), int(right)
                round_diff_total += (left - right)
            except:
                pass  # in case of missing data

    total_matches = len(recent)
    winrate = wins / total_matches if total_matches > 0 else 0
    avg_round_diff = round_diff_total / total_matches if total_matches > 0 else 0

    return {
        "recent_winrate": winrate,
        "avg_round_diff": avg_round_diff,
        "matches_used": total_matches
    }

features = compute_team_features(team_data)
print(features)

# Scraper Fully Put Together
# ------------------------------------------------------------------------------------------------------------------------

def get_match_features(match_url):
    teams = get_teams_from_match(match_url)
    match_features = {}

    for team in teams:
        history = get_team_history(get_completed_matches_url(team["url"]))
        features = compute_team_features(history)
        match_features[team["name"]] = features
    
    return match_features

res = get_match_features(matches[0]["link"])
print(res)

# ------------------------------------------------------------------------------------------------------------------------


def build_time_aware_training_dataset(max_pages=1, last_n=10):
    dataset = []

    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/results/?page={page}"
        print(f"Scraping results from {url}...")
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        rows = soup.select("a.wf-module-item.match-item")
        if not rows:
            break

        for row in rows:
            match_url = "https://www.vlr.gg" + row["href"]
            teams = get_teams_from_match(match_url)
            if len(teams) != 2:
                continue

            team_names = [t["name"] for t in teams]

            # Get match date from match page
            match_page = requests.get(match_url, headers=HEADERS)
            match_soup = BeautifulSoup(match_page.text, "html.parser")
            date_div = match_soup.select_one(".match-header-date .moment-tz-convert")
            match_dt = None
            if date_div and date_div.has_attr("data-utc-ts"):
                try:
                    match_dt = datetime.strptime(date_div["data-utc-ts"], "%Y-%m-%d %H:%M:%S")
                except Exception as e:
                    print("Error parsing match datetime:", e)

            # Determine winner
            winner_div = row.select_one(".match-item-vs-team.mod-winner .text-of")
            winner_name = winner_div.get_text(strip=True) if winner_div else None
            match_features = {
                "link": match_url,
                "winner": 0 if winner_name == team_names[0] else 1,
                "date": match_dt
            }

            for idx, t in enumerate(teams):
                full_history = get_team_history(get_completed_matches_url(t["url"]))
                # Only include matches before this match date
                past_matches = [
                    m for m in full_history
                    if m["datetime"] and match_dt and m["datetime"] < match_dt
                ]
                # Sort chronologically and take last_n
                past_matches = sorted(past_matches, key=lambda x: x["datetime"])[-last_n:]

                feats = compute_team_features(past_matches, last_n=last_n)
                prefix = "teamA" if t["name"] == team_names[0] else "teamB"
                for k, v in feats.items():
                    match_features[f"{prefix}_{k}"] = v

            dataset.append(match_features)

    return dataset



training_data = build_time_aware_training_dataset(max_pages=1, last_n=10)
print("Collected", len(training_data), "training samples")
print(training_data)
