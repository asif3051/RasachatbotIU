
#36e0b6dd87554ad49f2a1324cc992ae4

from typing import Any, Text, Dict, List, Callable, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from datetime import datetime 

class ActionHandleLeagueQuestion(Action):
    API_KEY = "36e0b6dd87554ad49f2a1324cc992ae4"
    BASE_URL = "https://api.football-data.org/v4"
    
    def name(self) -> Text:
        return "action_handle_league_question"

    def __init__(self):
        self.headers = {"X-Auth-Token": self.API_KEY}
        self.questions = self._initialize_questions()

    def _format_table(self, standings_data: Dict) -> str:
        """Format the complete league table."""
        table = standings_data['standings'][0]['table']
        result = "Premier League Table:\n"
        for team in table:
            result += (f"{team['position']}. {team['team']['name']} - "
                      f"Points: {team['points']}, "
                      f"Played: {team['playedGames']}, "
                      f"GD: {team['goalDifference']}\n")
        return result

    def _format_matches(self, matches_data: Dict) -> str:
        """Format upcoming matches."""
        matches = matches_data['matches'][:5]  # Show next 5 matches
        if not matches:
            return "No upcoming matches scheduled"
            
        result = "Upcoming Fixtures:\n"
        for match in matches:
            match_date = datetime.strptime(match['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
            result += (f"{match['homeTeam']['name']} vs {match['awayTeam']['name']}\n"
                      f"Date: {match_date.strftime('%Y-%m-%d %H:%M')} UTC\n\n")
        return result.strip()

    def _format_team_matches(self, matches_data: Dict, team_name: str) -> str:
        """Format specific team matches."""
        matches = [m for m in matches_data['matches'] if 
                  team_name.lower() in m['homeTeam']['name'].lower() or 
                  team_name.lower() in m['awayTeam']['name'].lower()]
        if not matches:
            return f"No upcoming matches found for {team_name}"
            
        next_match = matches[0]
        match_date = datetime.strptime(next_match['utcDate'], "%Y-%m-%dT%H:%M:%SZ")
        return (f"Next match for {team_name}:\n"
                f"{next_match['homeTeam']['name']} vs {next_match['awayTeam']['name']}\n"
                f"Date: {match_date.strftime('%Y-%m-%d %H:%M')} UTC")

    def _format_relegation_zone(self, standings_data: Dict) -> str:
        """Format relegation zone information."""
        table = standings_data['standings'][0]['table']
        relegation_teams = table[-3:]
        result = "Relegation Zone:\n"
        for team in relegation_teams:
            result += f"{team['position']}. {team['team']['name']} - {team['points']} points\n"
        return result

    def _format_champions_league(self, standings_data: Dict) -> str:
        """Format Champions League spots information."""
        table = standings_data['standings'][0]['table']
        cl_teams = table[:4]
        result = "Champions League Spots:\n"
        for team in cl_teams:
            result += f"{team['position']}. {team['team']['name']} - {team['points']} points\n"
        return result

    def _initialize_questions(self) -> Dict:
        """Initialize question mappings with their respective API endpoints and processors."""
        return {
            "pl_table": {
                "patterns": ["premier league table", "pl table", "league standings", "show table"],
                "endpoint": f"{self.BASE_URL}/competitions/PL/standings",
                "process": lambda d: self._format_table(d)
            },
            "team_next_match": {
                "patterns": ["when is", "next match", "when do", "playing next"],
                "endpoint": f"{self.BASE_URL}/competitions/PL/matches?status=SCHEDULED",
                "process": lambda d, team="": self._format_team_matches(d, team)
            },
            "relegation_zone": {
                "patterns": ["relegation zone", "bottom three", "who's going down"],
                "endpoint": f"{self.BASE_URL}/competitions/PL/standings",
                "process": lambda d: self._format_relegation_zone(d)
            },
            "champions_league": {
                "patterns": ["champions league spots", "top four", "ucl positions"],
                "endpoint": f"{self.BASE_URL}/competitions/PL/standings",
                "process": lambda d: self._format_champions_league(d)
            },
            "fixtures": {
                "patterns": ["fixtures", "upcoming matches", "next games", "match schedule", "this weekend"],
                "endpoint": f"{self.BASE_URL}/competitions/PL/matches?status=SCHEDULED",
                "process": lambda d: self._format_matches(d)
            },
            "top_scorer": {
                "patterns": ["top scorer", "golden boot", "most goals"],
                "endpoint": f"{self.BASE_URL}/competitions/PL/scorers",
                "process": lambda d: f"The top scorer is {d['scorers'][0]['player']['name']} with {d['scorers'][0]['goals']} goals"
            }
        }

    def _extract_team_name(self, message: str) -> Optional[str]:
        """Extract team name from user message."""
        common_teams = [
            "Arsenal", "Chelsea", "Liverpool", "Manchester City", "Manchester United",
            "Tottenham", "Newcastle", "Aston Villa", "West Ham", "Brighton"
        ]
        message = message.lower()
        for team in common_teams:
            if team.lower() in message:
                return team
        return None

    def _match_question(self, user_message: str) -> Optional[tuple]:
        """Match user message to predefined questions using patterns."""
        user_message = user_message.lower()
        team_name = self._extract_team_name(user_message)
        
        for q_id, q_info in self.questions.items():
            if any(pattern in user_message for pattern in q_info["patterns"]):
                return q_info, team_name
        return None, None

    def _fetch_and_process_data(self, endpoint: str, process_func: Callable, team_name: Optional[str] = None) -> str:
        """Fetch and process API data with error handling."""
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return process_func(data) if team_name is None else process_func(data, team_name)
        except requests.RequestException as e:
            raise ValueError(f"Could not fetch data from the server: {str(e)}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Could not find the requested information: {str(e)}")
        except Exception as e:
            raise ValueError(f"An error occurred while processing the data: {str(e)}")

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            user_message = tracker.latest_message.get("text", "")
            matched_question, team_name = self._match_question(user_message)
            
            if not matched_question:
                dispatcher.utter_message(
                    text="something went wrong, try again!"
                )
                return []

            answer = self._fetch_and_process_data(
                matched_question["endpoint"],
                matched_question["process"],
                team_name
            )
            dispatcher.utter_message(text=answer)
            
        except ValueError as e:
            dispatcher.utter_message(text=f"Sorry, I couldn't get that information: {str(e)}")
        except Exception as e:
            dispatcher.utter_message(text="I encountered an error while processing your request. Please try again.")
        
        return []


class ActionHandleGenericQuestion(Action):
    def name(self) -> Text:
        return "action_handle_generic_question"
    
    def __init__(self):
        # Load the FLAN-T5 Large model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        try:
            # Get the user's question from the tracker
            question = tracker.latest_message.get('text')
            
            # Predefined answers for common football questions
            common_answers = {
                "how many players each team have": "A football team consists of 11 players, including the goalkeeper. Teams also have substitutes on the bench.",
                "what are the dimensions of a football pitch": "The dimensions of a football pitch vary, but it is typically 100-110 meters in length and 64-75 meters in width.",
                "how big is a football ground": "A standard football pitch is about 100-110 meters long and 64-75 meters wide.",
                "who is a goalkeeper": "A goalkeeper is the player whose main role is to prevent the opposing team from scoring by blocking shots at the goal. The goalkeeper is the only player allowed to use their hands within the penalty area.",
                "what is offside in football": "In football, a player is considered offside if they are closer to the opponent's goal than both the ball and the second last defender when the ball is played to them, except when they are in their own half.",
                "how is a penalty kick taken": "A penalty kick is awarded when a player commits a foul inside their own penalty area. The ball is placed on the penalty spot, 11 meters from the goal line, and the player takes a shot at goal with only the goalkeeper to defend.",
                "what is a corner kick": "A corner kick is awarded when the ball goes over the goal line, last touched by a player from the defending team. The ball is placed in one of the four corner arcs and kicked into play by the attacking team.",
                "how many halves in a football match": "A football match consists of two halves, each lasting 45 minutes, with a 15-minute halftime break in between.",
                "what is the role of a referee in football": "The referee is responsible for enforcing the rules of football, ensuring fair play, and making decisions such as awarding fouls, goals, and penalties.",
                "what is a red card": "A red card is shown to a player for serious foul play or unsporting behavior. It results in the player being sent off the field and not being allowed to return to the match.",
                "what is a yellow card": "A yellow card is shown as a caution to a player for unsporting behavior or a minor foul. If a player receives two yellow cards in one game, they are sent off with a red card."
            }
            
            # Check if the question matches any predefined answer
            for key, answer in common_answers.items():
                if key.lower() in question.lower():
                    dispatcher.utter_message(text=answer)
                    return []
            
            # Create the prompt for the model
            prompt = (f"Explain the following football concept in detail and provide examples: {question}. "
            "Make the explanation beginner-friendly.")
            
            # Tokenize the input and generate a response
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=100,  # Adjust based on desired response length
                num_return_sequences=3,  # Generate multiple answers
                do_sample=True,
                num_beams=3,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=2
            )
            
            # Decode the responses and select the best one
            answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            best_answer = max(answers, key=lambda ans: len(ans.split()))  # Select the longest, more detailed answer
            
            # Send the best response to the user
            dispatcher.utter_message(text=best_answer.strip())
        
        except Exception as e:
            dispatcher.utter_message(text=f"I encountered an issue: {str(e)}")
        
        return []

# class ActionHandleGenericQuestion(Action):
#     def name(self) -> Text:
#         return "action_handle_generic_question"
        
#     def __init__(self):
#         self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.context_sections = self._initialize_context()
#         self.embeddings = self._prepare_embeddings()
        
#         # Load Phi-2 model
#         self.model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
#         self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         try:
#             question = tracker.latest_message.get('text')
#             relevant_context = self._get_relevant_context(question)
            
#             prompt = f"Context: {relevant_context}\n\nQ: {question}\nA:"
#             inputs = self.tokenizer(prompt, return_tensors="pt")
            
#             outputs = self.model.generate(
#                 inputs.input_ids,
#                 max_length=100,
#                 top_p=0.9,
#                 temperature=0.3
#             )
            
#             answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#             answer = answer.split("Q:")[0].replace(prompt, "").strip()
            
#             dispatcher.utter_message(text=answer)
            
#         except Exception as e:
#             dispatcher.utter_message(text="I apologize, but I'm having trouble processing your question.")
            
#         return []
# class ActionHandleGenericQuestion(Action):
#     def name(self) -> Text:
#         return "action_handle_generic_question"
        
#     def __init__(self):
#         self.qa_model = pipeline(
#             "question-answering",
#             model="distilbert-base-cased-distilled-squad",
#             tokenizer="distilbert-base-cased-distilled-squad"
#         )
#         self.context_sections = self._initialize_context_sections()
        
#     def _initialize_context_sections(self) -> Dict[str, str]:
#         """Initialize context sections with keywords for better matching"""
#         return {
#             "cards_and_discipline": """
#             Cards and Discipline in Football:
#             Yellow cards are warnings given for minor offenses or unsporting behavior.
#             Red cards result in the player being sent off and banned for at least one match.
#             Two yellow cards in the same game result in a red card.
#             Dangerous tackles, deliberate handballs, and violent conduct can result in direct red cards.
#             Players who receive a red card must leave the field immediately.
#             Accumulating multiple yellow cards across different games can lead to suspension.
#             """,
            
#             "basic_rules": """
#             Basic Football Rules:
#             Football (soccer) is played between two teams of 11 players each on the field.
#             The game lasts 90 minutes, split into two 45-minute halves with a 15-minute break.
#             Extra time can be added for injuries, substitutions, and time-wasting.
#             Teams can make up to 5 substitutions per game in most competitions.
#             The objective is to score goals by getting the ball into the opposing team's goal.
#             A match is won by the team scoring the most goals.
#             """,
            
#             "offside_rule": """
#             The Offside Rule:
#             A player is offside if they are closer to the opponent's goal than the ball and the second-last defender when the ball is played.
#             Players cannot be offside from goal kicks, throw-ins, or corner kicks.
#             Being in an offside position is not an offense unless the player becomes involved in active play.
#             A player in an offside position receiving the ball from an opponent is not offside.
#             The offside position is judged at the moment the ball is played by a teammate.
#             """,
            
#             "scoring_and_points": """
#             Scoring and Points System:
#             Teams get 3 points for a win and 1 point for a draw in league competitions.
#             No points are awarded for a loss.
#             The team with the most points at the end of the season wins the league.
#             Goal difference is used as a tiebreaker when teams have equal points.
#             Head-to-head records may also be used to separate teams level on points.
#             """,
            
#             "set_pieces": """
#             Set Pieces in Football:
#             Corner kicks are awarded when the ball goes over the goal line off a defending player.
#             Throw-ins are taken when the ball goes out over the sideline.
#             Direct free kicks can be scored directly, while indirect free kicks must touch another player.
#             Penalty kicks are awarded for fouls committed inside the penalty area.
#             The ball must be stationary before taking any set piece.
#             """,
            
#             "goalkeeper_rules": """
#             Goalkeeper Rules:
#             Goalkeepers can handle the ball within their own penalty area.
#             They cannot pick up deliberate back-passes from teammates using their feet.
#             Goalkeepers must release the ball within six seconds of gaining control.
#             The goalkeeper can be sent off and shown a red card like any other player.
#             Goalkeepers can only handle the ball inside their penalty area.
#             """,
            
#             "match_officials": """
#             Match Officials:
#             Each match has one referee and two assistant referees (linesmen).
#             A fourth official manages substitutions and indicates added time.
#             VAR (Video Assistant Referee) can be used to check goals, penalties, red cards, and mistaken identity.
#             The referee's decision is final for all on-field matters.
#             Additional assistant referees may be used in some competitions.
#             """
#         }
    
#     def _get_relevant_context(self, question: str) -> str:
#         """Select relevant context section based on question keywords"""
#         question = question.lower()
        
#         # Define keyword mappings
#         keyword_mappings = {
#             "cards_and_discipline": ["card", "yellow", "red", "sent off", "warning", "suspension", "foul"],
#             "basic_rules": ["substitutions", "player", "game", "time", "break", "half", "duration", "long"],
#             "offside_rule": ["offside", "position"],
#             "scoring_and_points": ["point", "score", "win", "draw", "loss", "league"],
#             "set_pieces": ["corner", "free kick", "penalty", "throw"],
#             "goalkeeper_rules": ["goalkeeper", "goalie", "keeper", "handle", "back pass"],
#             "match_officials": ["referee", "var", "official", "linesman", "assistant"]
#         }
        
#         # Find matching sections
#         relevant_sections = []
#         for section, keywords in keyword_mappings.items():
#             if any(keyword in question for keyword in keywords):
#                 relevant_sections.append(self.context_sections[section])
        
#         # If no specific section matches, combine all sections
#         if not relevant_sections:
#             return " ".join(self.context_sections.values())
        
#         # Return combined relevant sections
#         return " ".join(relevant_sections)

#     def get_full_sentence(self, answer: str, context: str) -> str:
#         """Extract full sentence containing the answer with improved matching"""
#         # Split into sentences and clean them
#         sentences = [s.strip() for s in context.split('.') if s.strip()]
        
#         # Try exact matching first
#         for sentence in sentences:
#             if answer.lower() in sentence.lower():
#                 return sentence + "."
                
#         # If no exact match, try fuzzy matching
#         best_match = None
#         highest_overlap = 0
#         answer_words = set(answer.lower().split())
        
#         for sentence in sentences:
#             sentence_words = set(sentence.lower().split())
#             overlap = len(answer_words & sentence_words)
#             if overlap > highest_overlap:
#                 highest_overlap = overlap
#                 best_match = sentence
        
#         return (best_match + ".") if best_match else answer

#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         try:
#             question = tracker.latest_message.get('text')
            
#             # Get relevant context section
#             relevant_context = self._get_relevant_context(question)
            
#             # Get answer using the model
#             result = self.qa_model(question=question, context=relevant_context)
            
#             if result['score'] > 0.3:  # Increased confidence threshold
#                 full_answer = self.get_full_sentence(result['answer'], relevant_context)
                
#                 # Additional validation to ensure answer is relevant
#                 if len(full_answer.split()) > 3:  # Avoid very short answers
#                     dispatcher.utter_message(text=full_answer)
#                     return []
            
#             # If no good answer found, provide a more specific response
#             dispatcher.utter_message(text="I'm not sure about that specific question. Could you rephrase it or ask something more specific about football rules?")
                
#         except Exception as e:
#             dispatcher.utter_message(text="I apologize, but I'm having trouble understanding your question. Could you rephrase it?")
            
#         return []