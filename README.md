# Parallel Monte Carlo Tree Search with MPI and Groq

> Enhancing SLM responses using distributed Monte Carlo Tree Search

## Overview

This project implements a novel approach to improving LLM responses using a parallel Monte Carlo Tree Search (MCTS) algorithm powered by MPI (Message Passing Interface). By distributing the search process across multiple cores, we can efficiently explore and evaluate different response variations to find optimal answers.

## Features

- **Parallel MCTS Implementation**: Utilizes MPI for distributed processing
- **Groq Integration**: Leverages llama-3.2-1b-preview for response generation and evaluation
- **Adaptive Response Improvement**: Iteratively enhances answers through tree exploration
- **Detailed Logging**: Comprehensive logging system using Loguru
- **Score-based Evaluation**: Quantitative assessment of response quality

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/parallel-mcts-llm.git
cd parallel-mcts-llm
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install mpi4py openai numpy loguru
```

4. Set up your OpenAI API key:

```bash
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_API_BASE_URL=https://api.groq.com/openai/v1
```

## Usage

Run the program with MPI:

```bash
mpiexec -n 4 python app.py
```

The `-n 4` parameter specifies the number of processes:

- Process 0: Root process (manages the search tree)
- Processes 1-3: Worker processes (evaluate responses)

## How It Works

1. **Initialization**

   - Starts with a basic "I don't know" response
   - Creates initial tree structure

2. **MCTS Process**

   - **Selection**: Choose promising nodes using UCT
   - **Expansion**: Generate new response variations
   - **Simulation**: Evaluate responses in parallel
   - **Backpropagation**: Update scores through the tree

3. **Parallel Processing**
   ```
   Root Process (0)               Worker Processes (1-3)
   ├── Manages tree              ├── Rate responses
   ├── Coordinates search        ├── Evaluate variations
   └── Collects results          └── Send scores back
   ```

## Performance

The parallel implementation offers significant advantages:

- Evaluates multiple responses simultaneously
- Reduces total processing time
- Scales with available cores

## Configuration

Key parameters that can be tuned:

- `num_iterations`: Number of MCTS iterations
- `C`: Exploration constant in UCT calculation
- Number of parallel processes

## Example

```python
Question: "What is quantum computing?"

Initial Response: "I don't know"

Final Response: "Quantum computing is a type of computation that harnesses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously. This allows quantum computers to solve certain problems exponentially faster than classical computers, particularly in areas like cryptography, drug discovery, and complex optimization problems."

Score: 92/100
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Future Improvements

- [ ] Variability and error in JSON responses
- [ ] Add API support for monte carlo
- [ ] Integrate with other LLM providers
- [ ] Add visualization of the search tree
- [ ] Implement caching for API calls
- [ ] Add unit tests and benchmarks

## References

- Monte Carlo Tree Search: [Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- MPI for Python: [mpi4py Documentation](https://mpi4py.readthedocs.io/)
