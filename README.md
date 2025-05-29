## Basic_Reinforcement_Learning

This project implements a simple Reinforcement Learning agent solving a grid world environment using policy iteration.

# Features

>Grid world with customizable size and goal state

>Rewards: -1 for all states except the goal state with reward 1

>Agent starts with a random policy

>Uses discounted rewards with configurable discount factor (lambda)

>Iterative policy evaluation and improvement until convergence

>Prints policy as arrows representing actions in each state

>Prints value function for each state during iterations


## Usage

1. Compile the C++ code
   
    g++ -std=c++11 -o rl_grid  RLAgent.cpp

2. Run the program

    ./rl_grid

## Customize

Modify grid size, goal position, or discount factor by editing the main() function.

## Output

-- Policy displayed with arrows (^, v, <, >) indicating actions per state

-- Goal state marked with G

-- State value table showing estimated discounted rewards
