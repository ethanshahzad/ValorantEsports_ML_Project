# Valorant Esports ML Project

## Overview:
This project leverages machine learning to predict Valorant esports match outcomes. By analyzing team statistics and integrating AI-generated confidence scores, the model aims to provide accurate predictions for the match winner.

## Features:
- Data Collection: Scrapes and processes hundreds of match data from the vlr.gg website.
- Feature Engineering: Incorporates team statistics such as win rates, round differences, win streaks, and recent form.
- Modeling: Utilizes XGBoost for classification tasks.
- AI Integration: Employs OpenAI's GPT to generate confidence scores for predictions.

## Folder Structure
- `scraper/` – Contains `vlrScraper.py` to collect match data.
- `models/` – ML models using traditional features like diff columns.
- `LLM model/` – Machine learning model using GPT confidence scores and further model optimization.

## Running the Program:
- Download all project files.
- Run **vlrScraper.py** in the scraper folder to fetch data from the most recent completed matches.  
  - This generates a `dataset.csv` file.
- Move `dataset.csv` to the `models/` and `LLM model/` folders.
- Begin the classification process in each folder by running its respective py file.

## Conclusion:
After finalizing optimizations, 70% accuracy was achieved (accuracy may vary depending on random split). While these results are promising, there is potential to improve the model further, such as by incorporating head-to-head matchups. Contributions are welcome!
