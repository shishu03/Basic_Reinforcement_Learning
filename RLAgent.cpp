#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <map>

enum Action { UP, DOWN, LEFT, RIGHT };
const std::map<Action, char> action_symbols = {
    {UP, '^'}, {DOWN, 'v'}, {LEFT, '<'}, {RIGHT, '>'}
};

struct State {
    int row, col;
};

class GridWorld {
public:
    int rows, cols;
    State goal;
    double lambda;  // discount factor
    std::vector<std::vector<double>> rewards;
    std::vector<std::vector<Action>> policy;
    std::vector<std::vector<double>> values;

    GridWorld(int r, int c, State g, double l) :
        rows(r), cols(c), goal(g), lambda(l),
        rewards(r, std::vector<double>(c, -1.0)),
        policy(r, std::vector<Action>(c)),
        values(r, std::vector<double>(c, 0.0))
    {
        // Set reward for goal
        rewards[goal.row][goal.col] = 1.0;
        initialize_random_policy();
    }

    void initialize_random_policy() {
        std::srand(std::time(nullptr));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (i == goal.row && j == goal.col) {
                    // No action needed at goal (can keep any)
                    policy[i][j] = UP;
                    values[i][j] = rewards[i][j];
                } else {
                    policy[i][j] = static_cast<Action>(std::rand() % 4);
                }
            }
        }
    }

    State next_state(const State& s, Action a) {
        State ns = s;
        switch (a) {
            case UP:    ns.row = std::max(0, s.row - 1); break;
            case DOWN:  ns.row = std::min(rows - 1, s.row + 1); break;
            case LEFT:  ns.col = std::max(0, s.col - 1); break;
            case RIGHT: ns.col = std::min(cols - 1, s.col + 1); break;
        }
        return ns;
    }

    void evaluate_policy(double theta=1e-4, int max_iter=1) {
        // Policy evaluation using iterative method
        for (int iter = 0; iter < max_iter; ++iter) {
            double delta = 0.0;
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if (i == goal.row && j == goal.col) continue; // goal value fixed
                    State s{i, j};
                    Action a = policy[i][j];
                    State ns = next_state(s, a);
                    double v = rewards[i][j] + lambda * values[ns.row][ns.col];
                    delta = std::max(delta, std::abs(v - values[i][j]));
                    values[i][j] = v;
                }
            }
            if (delta < theta) break;
        }
    }

    bool improve_policy() {
        // Policy improvement; returns true if policy changed
        bool policy_stable = true;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (i == goal.row && j == goal.col) continue; // goal policy fixed
                State s{i, j};
                Action old_action = policy[i][j];
                double best_value = -1e9;
                Action best_action = old_action;

                // Try all actions to find best
                for (int a = 0; a < 4; ++a) {
                    Action act = static_cast<Action>(a);
                    State ns = next_state(s, act);
                    double v = rewards[i][j] + lambda * values[ns.row][ns.col];
                    if (v > best_value) {
                        best_value = v;
                        best_action = act;
                    }
                }

                policy[i][j] = best_action;
                if (best_action != old_action) policy_stable = false;
            }
        }
        return policy_stable;
    }

    void print_policy() {
        std::cout << "Policy:\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (i == goal.row && j == goal.col) {
                    std::cout << " G ";
                } else {
                    std::cout << " " << action_symbols.at(policy[i][j]) << " ";
                }
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    void print_values() {
        std::cout << "State Values:\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%6.2f ", values[i][j]);
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

int main() {
    int rows = 10, cols = 10;
    State goal = {1, 3};
    double lambda = 0.9;

    GridWorld env(rows, cols, goal, lambda);
    env.print_policy();

    int iteration = 0;
    while (true) {
        iteration++;
        env.evaluate_policy();
        bool stable = env.improve_policy();
        std::cout << "Iteration " << iteration << ":\n";
        env.print_policy();
        env.print_values();
        if (stable) break;
    }

    std::cout << "Optimal policy found after " << iteration << " iterations.\n";

    return 0;
}
