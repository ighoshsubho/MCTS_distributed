from mpi4py import MPI
from openai import OpenAI
import os
from dotenv import load_dotenv
from loguru import logger
load_dotenv()

from helper import Node, mcts_iteration

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Add rank to logger context
    logger.configure(extra={"rank": rank})
    
    logger.info(f"Process {rank} started")

    try:
        client = OpenAI(api_key=os.getenv('GROQ_API_KEY'), 
                        base_url=os.getenv('GROQ_API_BASE_URL'))
        logger.debug("OpenAI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        comm.Abort()
        return

    if rank == 0:
        # Get the question from user
        question = input("Enter your question: ")
        logger.info(f"Question received: {question}")
        
        # Initialize MCTS with "I don't know" as root
        root = Node("I don't know")
        root.visits = 1
        
        num_iterations = 3  # Adjust based on desired depth/breadth
        for i in range(num_iterations):
            logger.info(f"Starting MCTS iteration {i+1}")
            mcts_iteration(client, question, root, comm, rank, size)
        
        # Find best path from root to leaf
        current = root
        path = [current]
        while current.children:
            current = max(current.children, key=lambda x: x.total_score/x.visits if x.visits > 0 else 0)
            path.append(current)
        
        # Print results
        print("\nMCTS Search Results:")
        print(f"Question: {question}")
        print("\nAnswer evolution:")
        for i, node in enumerate(path):
            print(f"\nIteration {i}:")
            print(f"Answer: {node.response}")
            print(f"Score: {node.total_score/node.visits if node.visits > 0 else 0:.2f}")
            print(f"Visits: {node.visits}")
        
        print(f"\nBest final answer: {path[-1].response}")
        print(f"Final score: {path[-1].total_score/path[-1].visits if path[-1].visits > 0 else 0:.2f}")
        
    else:
        # Worker processes participate in MCTS iterations
        num_iterations = 3
        for _ in range(num_iterations):
            mcts_iteration(client, "", root=None, comm=comm, rank=rank, size=size)

    logger.info(f"Process {rank} finished")

if __name__ == "__main__":
    main()