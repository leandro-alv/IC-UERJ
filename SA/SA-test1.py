#! ~/IC-UERJ/environment/python3

import random

EE = 2.71828


def anneal(solution):

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
            if ap > random():
                solution = new_solution
                old_cost = new_cost
            i += 1
        T = T * ALPHA

    return solution, old_cost


def cost(solution):

    cost = 0
    alm_count = 0
    fst_len = len(solution[0])
    snd_len = len(solution[1])
    min_len = min(fst_len, snd_len)
    max_len = max(fst_len, snd_len)

    for i in range(min_len):
        if solution[0][i] == solution[1][i]:
            alm_count += 1

    max_spaces = max(solution[0].count(" "), solution[1].count(" "))
    cost = ((alm_count * 2.0) - max_spaces) / 3.0

    return cost


def neighbor(solution):

    

    return solution


def acceptance_probability(old_cost, new_cost, T):

    ap = EE * ((old_cost - new_cost) / T)

    return ap


def main():

    init_solution = ["AABDC",
                     "ABC"]
    best_solution = anneal(init_solution)

    print(best_solution)


if __name__ == "__main__":
    main()


'''
ptr = 0
    for i in range(len(solution[0])):
        for j in range(ptr, len(solution[1])):
            if solution[0][i] == solution[1][j]:
                if i != j:
                    if i < j:
                        # Shift solution[0][i] to the right
                        solution[0] = solution[0][:i] + " " + solution[0][i:]
                    else:
                        # Shift solution[0][j] to the right
                        solution[1] = solution[1][:j] + " " + solution[1][j:]
                    return solution
                else:
                    ptr += 1
                    break
'''