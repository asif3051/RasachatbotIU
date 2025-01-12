# Rasa Football Chatbot
## Overview
This project is a football assistant chatbot built using Rasa. The bot can answer questions about football leagues, teams, fixtures, and general football knowledge. It utilizes Rasa NLU (Natural Language Understanding) to process user queries and Rasa Actions to fetch and provide data from a football API.
## Features
- Greet and Goodbye interactions
- Retrieve football league standings
- Show upcoming fixtures for teams
- Get information about relegation zones, champions league spots, top scorers, and more
- Answer general football-related questions like "What is offside?" or "How many players does a team have?"
## Prerequisites
- Python 3.6 > 3.9 
- Rasa (Install via pip install rasa)
## Installation and Setup
1] Create a virtual environment in Python:
- python -m venv .\venv 

2] Activate the virtual environment:
- .\venv\Scripts\activate

3] Train the Rasa model:
- rasa train

**Split the terminal in two**

4] Run the action server in one terminal:
- rasa run actions

5] Run the Rasa chatbot in other terminal:
- rasa shell
## How to Interact with the Chatbot
Once the chatbot is running, you can start chatting with it by asking football-related questions like:
- "What's the Premier League table?"
- "When is Arsenal's next match?"
- "What is offside in football?"
- "Who's in the relegation zone?"
## Actions
*ActionHandleLeagueQuestion*
- Fetches data from the Football Data API for league standings, fixtures, top scorers, etc.
- Processes responses like upcoming matches, relegation zones, etc.

*ActionHandleGenericQuestion*
- Uses a pre-trained FLAN-T5 model to answer general football-related questions.
- Provides detailed explanations for terms like offside, penalty kicks, and more.