#! ~/IC-UERJ/environment/python3

import re
import operator
from itertools import zip_longest, combinations
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


def column_cost(column):
    """Calculates the cost of the column using the sum-of-pairs(SP).
    Each match is +1, mismatch -1 and gap -2. Two gaps is 0, to avoid extra
    penalization.
    """
    sp = 0
    for comb in combinations(column, 2):
        if operator.eq(comb[0], comb[1]):
            if operator.ne(comb[0], '-'):
                sp += 1
        else:
            if operator.ne(comb[0], '-') and operator.ne(comb[1], '-'):
                sp -= 1
            else:
                sp -= 2
    return sp


def cost(solution):
    """Calculates the cost of the solution using the sum-of-pairs(SP) for each
    column.
    """
    return sum(map(column_cost, zip_longest(*solution, fillvalue='-')))


def neighbor(solution):
    """Generates a new random solution based on a previous one by adding
    spaces. There's a 50% probability of adding space in one of the
    parts of the solution.
    """
    min_len = min(map(len, solution))
    min_sol = list(filter(lambda x: len(x) == min_len, solution))
    if operator.ne(len(solution), len(min_sol)):
        index = solution.index(min_sol[0])
        i = random.randint(0, len(solution[index]))
        solution[index] = solution[index][:i] + "-" + solution[index][i:]
    else:
        index = random.randint(0, len(solution) - 1)
        if index:
            occur = [m.start() for m in re.finditer('-', solution[index])]
            i = random.choice(occur)
            sol_list = list(solution[index])
            if i == 0:
                sol_list[i] = sol_list[i + 1]
                sol_list[i + 1] = '-'
            elif i == (len(solution[index]) - 1):
                sol_list[i] = sol_list[i - 1]
                sol_list[i - 1] = '-'
            else:
                if random.random() >= 0.5:
                    sol_list[i] = sol_list[i + 1]
                    sol_list[i + 1] = '-'
                else:
                    sol_list[i] = sol_list[i - 1]
                    sol_list[i - 1] = '-'
            solution[index] = "".join(sol_list)

    return solution


def acceptance_probability(old_cost, new_cost, T):
    """Calculates the acceptance probability which is the recommendation on
    whether or not to jump to the new solution.
    """
    return EE * ((new_cost - old_cost) / T)


def main():
    init_solution = ["AABDC", "ABC", "AC"]
    best_solution = anneal(init_solution)

    print(best_solution)


if __name__ == "__main__":
    main()
