import json
import math
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger


@dataclass
class Node:
    response: str
    visits: int = 0
    total_score: float = 0
    parent: Optional['Node'] = None
    children: List['Node'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def uct(self):
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_score / self.visits
        exploration = math.sqrt(2 * math.log(self.parent.visits) / self.visits) if self.parent else 0
        return exploitation + exploration
    
def get_suggestions(client, question, response):
    logger.info(f"Getting suggestions for response to question: '{question}'")
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-1b-preview",
            messages=[
                {"role": "system", "content": "You are helping improve answer quality. Generate exactly 3 specific suggestions for improving the answer. Format as JSON format like this - {'suggestions': ['Suggestion 1', 'Suggestion 2', 'Suggestion 3']}."},
                {"role": "user", "content": f"Question: {question}\nCurrent answer: {response}\nHow can this answer be improved?"}
            ],
            response_format={ "type": "json_object" }
        )
        suggestions = json.loads(completion.choices[0].message.content)
        logger.success(f"Received suggestions: {suggestions}")
        return suggestions['suggestions']
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return ["Add more specific details", "Include relevant examples", "Explain the reasoning"]

def generate_improved_responses(client, question, current_response, suggestions):
    logger.info("Generating improved responses")
    try:
        prompt = f"Question: {question}\nCurrent answer: {current_response}\n\nImprovement suggestions:\n"
        for i, sugg in enumerate(suggestions, 1):
            prompt += f"{i}. {sugg}\n"
        
        completion = client.chat.completions.create(
            model="llama-3.2-1b-preview",
            messages=[
                {"role": "system", "content": "Generate 3 improved answers based on the suggestions. Format as JSON format like this - {'responses': ['Response 1', 'Response 2', 'Response 3']}."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        responses = json.loads(completion.choices[0].message.content)['responses']
        logger.success(f"Generated {len(responses)} responses")
        for i, resp in enumerate(responses, 1):
            logger.debug(f"Response {i}: {resp}")
        return responses[:3]
    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        return [
            f"Based on {current_response}, I can elaborate...",
            f"While starting from {current_response}, we can add...",
            f"Building on {current_response}, consider..."
        ]

def rate_response(client, question, response):
    logger.info(f"Rating response for question: '{question}'")
    try:
        completion = client.chat.completions.create(
            model="llama-3.2-1b-preview",
            messages=[
                {"role": "system", "content": "Rate how well the answer addresses the question from 0-100. Consider accuracy, completeness, and clarity. Return only the numeric score."},
                {"role": "user", "content": f"Question: {question}\nAnswer: {response}\nRate (0-100):"}
            ]
        )
        rating = float(completion.choices[0].message.content.strip())
        rating = min(max(rating, 0), 100)
        logger.success(f"Response rated: {rating}/100")
        return rating
    except Exception as e:
        logger.error(f"Error rating response: {e}")
        return 50.0
    
def select_best_child(node):
    if not node.children:
        return None
    return max(node.children, key=lambda x: x.uct)

def mcts_iteration(client, question, root, comm, rank, size):
    if rank == 0:
        # Selection
        current = root
        while current.children and len(current.children) == 3:
            current = select_best_child(current)
        
        # Expansion
        if current.visits > 0 and len(current.children) < 3:
            suggestions = get_suggestions(client, question, current.response)
            new_responses = generate_improved_responses(client, question, current.response, suggestions)
            
            for response in new_responses:
                child = Node(response, parent=current)
                current.children.append(child)
            
            # Distribute work to workers
            for i in range(1, size):
                if i-1 < len(new_responses):
                    comm.send((question, new_responses[i-1]), dest=i, tag=1)
                else:
                    comm.send(None, dest=i, tag=1)
            
            # Collect ratings
            for i in range(1, min(len(new_responses) + 1, size)):
                rating = comm.recv(source=i, tag=2)
                current.children[i-1].total_score = rating
                current.children[i-1].visits = 1
            
            # Backpropagate
            while current:
                current.visits += 1
                current = current.parent
    else:
        # Worker processes
        data = comm.recv(source=0, tag=1)
        if data is not None:
            question, response = data
            rating = rate_response(client, question, response)
            comm.send(rating, dest=0, tag=2)