#! ~/IC-UERJ/environment/python3

import random

EE = 2.71828


def anneal(solution):
    """Core of the simulated annealing algorithm."""
    old_cost = cost(solution)
    T = 1.0
    T_min = 0.00001
    ALPHA = 0.9

    while T > T_min:
        i = 1
        while i <= 100:
            new_solution = neighbor(solution)
            new_cost = cost(new_solution)
            ap = acceptance_probability(old_cost, new_cost, T)
            if ap > random.random():
                solution = new_solution
                old_cost = new_cost
            i += 1
        T = T * ALPHA

    return solution, old_cost


def cost(solution):
    """Calculates the cost of the solution using the weighted average between
    operand/operator alignment and the max number of spaces.
    """
    cost = 0
    alm_count = 0  # alignment operand/operator count
    fst_len = len(solution[0])
    snd_len = len(solution[1])
    min_len = min(fst_len, snd_len)

    for i in range(min_len):
        if solution[0][i] == solution[1][i]:
            alm_count += 1

    max_spaces = max(solution[0].count(" "), solution[1].count(" "))
    cost = ((alm_count * 2.0) + max_spaces) / 3.0

    return cost


def neighbor(solution):
    """Generates a new random solution based on a previous one by adding
    spaces. There's a 50% probability of adding space in one of the
    parts of the solution.
    """
    fst_len = len(solution[0])
    snd_len = len(solution[1])
    if fst_len < snd_len:
        i = random.randint(0, len(solution[0])-1)
        solution[0] = solution[0][:i] + " " + solution[0][i:]
    elif fst_len > snd_len:
        i = random.randint(0, len(solution[1])-1)
        solution[1] = solution[1][:i] + " " + solution[1][i:]

    return solution


def acceptance_probability(old_cost, new_cost, T):
    """Calculates the acceptance probability which is the recommendation on
    whether or not to jump to the new solution.
    """
    ap = EE * ((new_cost - old_cost) / T)

    return ap


def main():
    init_solution = ["AABDC", "ABC"]
    best_solution = anneal(init_solution)

    print(best_solution)


if __name__ == "__main__":
    main()
