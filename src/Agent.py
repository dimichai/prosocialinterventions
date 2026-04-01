import json
import random

from pydantic import BaseModel

from openai import OpenAI
from openai.types.chat import ParsedChoice

import prompts as P

class Action(BaseModel):
    option: int
    content: str
    explanation: str

class BooleanAction(BaseModel):
    choice: str
    explanation: str

class Agent():

    def __init__(self, model: str, persona: dict = None):

        self.persona = persona

        self.llm = None
        self.model = model

        self.identifier = 0
        self.followers = 0

        self.used_tokens_input = 0
        self.used_tokens_output = 0
        self.used_tokens_cached = 0

    def __repr__(self):
        return f"User {self.identifier} with {self.followers} followers"
    
    def __str__(self):
        return f"User {self.identifier} with {self.followers} followers"

    def _generate_persona(self, persona_path: str) -> str:
        """
        From a list of personas, randomly select one to use as the agent's persona.
        Not used anymore due to the persona being passed as an argument to enforce consistency.
        """
        
        persona_list = json.load(open(persona_path, 'r'))
        return random.choice(persona_list)
    
    def _generate_sys_msg(self) -> str:
        """
        Generate a system message to introduce the agent to the system and its persona.
        """

        return P.AGENT_SYSTEM_MESSAGE.format(persona=self.persona["persona"])
    
    def _add_bio(self):

        prompt = P.AGENT_BIOGRAPHY_PROMPT.format(persona=self.persona["persona"])

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
            ]
        )

        self.persona['biography'] = response.choices[0].message.content

        # print(self.persona['biography'])
    
    def set_client(self, client: OpenAI):
        """
        Set the client for the agent to use for the simulation.
        """

        self.llm = client

    def refresh_client(self, new_client: OpenAI):
        """
        Refresh the client for the agent to use for the simulation.
        """

        self.llm.close()
        self.set_client(new_client)

    def json(self, include_persona: bool = False):
        """
        Return the agent's data in JSON format.
        """
        
        result = {
            "identifier": self.identifier,
            "followers": self.followers,
            "used_tokens_input": self.used_tokens_input,
            "used_tokens_output": self.used_tokens_output,
            "used_tokens_cached": self.used_tokens_cached
        }

        if include_persona:
            result['persona'] = self.persona

        return result
    
    def increase_followers(self):
        """
        Increase the number of followers by 1.
        """
        self.followers += 1
    
    def get_response(self, message: str, response_format = None) -> ParsedChoice:
        """
        Get the response from the agent to the given message.
        """

        response = self.llm.beta.chat.completions.parse(
            model=self.model,
            messages = [
                {"role": "system", "content": self._generate_sys_msg()},
                {"role": "user", "content": message}
            ],
            response_format=response_format

        )

        # Keep track of the tokens used for cost analysis
        self.used_tokens_input += response.usage.prompt_tokens
        self.used_tokens_output += response.usage.completion_tokens
        self.used_tokens_cached += response.usage.prompt_tokens_details.cached_tokens
        
        return response.choices[0]
    
    def link_with_user(self, other_agent: 'Agent', post_content: str, other_agent_posts: list, use_bio: bool = False,
                       use_follower_count: bool = True) -> str:
        """
        Supply the bio of another agent and let the user decide if they want to follow them.
        """

        recent_posts = "".join(str(post["post_content"]) + "\n\n" for post in other_agent_posts[:5])

        msg = P.FOLLOW_DECISION.format(
            post_content=post_content,
            user_id=other_agent.identifier,
            follower_count_line=f"Followers: {other_agent.followers}\n" if use_follower_count else "",
            bio_line=f"Bio: {other_agent.persona['biography']}\n" if use_bio else "",
            recent_posts=recent_posts,
        )

        response = self.get_response(msg, BooleanAction).message.parsed

        return True if response.choice.lower() == 'yes' else False, response.explanation
    
    def perform_action(self, news_data: list, timeline: list) -> Action:
        """
        The user is presented with a set of options to choose from based on their persona.
        - Repost a post from the timeline
        - Share a news headline with a comment
        - Do nothing
        """

        msg = P.PERFORM_ACTION_INSTRUCTIONS
        msg += P.PERFORM_ACTION_TIMELINE_HEADER

        for post in timeline:
            msg += str(post["post_content"])
            msg += "\n\n"

        msg += P.PERFORM_ACTION_NEWS_HEADER

        for i, news_item in enumerate(news_data, start=1):
            msg += f"ID: {i}\nTitle: {news_item['headline']}\nCategory: {news_item['category']}\nDescription: {news_item['short_description']}\n\n"

        # Get response and handle the action

        try:
            response = self.get_response(msg, response_format=Action)
        except Exception as e:
            print(f"Error: {e}")
            return Action(option=-1, content="", explanation=str(e)), msg

        parsed = response.message.parsed
        if parsed is None:
            print(f"Warning: Failed to parse action response. Raw content: {response.message.content}")
            return Action(option=-1, content="", explanation="Failed to parse response"), msg

        return parsed, msg
