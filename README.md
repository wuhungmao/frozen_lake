# Frozen Lake Solver

This project implements a reinforcement learning agent to solve the Frozen Lake environment from OpenAI Gym.  It uses value iteration to find the optimal policy.

## Installation

1.  Clone the repository: `git clone https://github.com/wuhungmao/frozen_lake.git`
2.  Navigate to the project directory: `cd frozen-lake-solver`
3.  Create a virtual environment (recommended): `python3 -m venv .venv`
4.  Activate the virtual environment: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
5.  Install the required dependencies: `pip install -r requirements.txt`

## File Structure

*   `main.py`: The main script that runs the value iteration algorithm and extracts the policy.
*   `frozen_lake.py`: Contains the implementation of the Frozen Lake environment wrapper and the value iteration algorithm.
*   `test_frozen_lake.py`: Contains the unit tests for the code.
*   `requirements.txt`: Lists the project's dependencies (gym, numpy, etc.).

## Usage

To run the main program:

```bash
python3 main.py --map_size 8 --is_slippery