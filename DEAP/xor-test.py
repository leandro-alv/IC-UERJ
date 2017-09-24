#! ~/IC-UERJ/environment/python3

import random
import operator

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt
import networkx as nx

# Initialize Xor problem input and output vectors

XOR_ENTRIES = 5
XOR_SIZE_M = 2 ** XOR_ENTRIES

# Matrix 32 x 5 (5 entries, 32 possibilities)
inputs = [[0] * XOR_ENTRIES for i in range(XOR_SIZE_M)]
outputs = [None] * XOR_SIZE_M

for i in range(XOR_SIZE_M):
    value = i
    divisor = XOR_SIZE_M
    for j in range(XOR_ENTRIES):
        divisor /= 2
        if value >= divisor:
            inputs[i][j] = 1
            value -= divisor
        else:
            inputs[i][j] = 0

    count_1s = inputs[i].count(1)
    outputs[i] = 1 if (count_1s > 0) and (count_1s % 2 != 0) else 0

pset = gp.PrimitiveSet("MAIN", XOR_ENTRIES, "IN")
pset.addPrimitive(operator.xor, 2)

creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMaxMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=2, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalXor(individual):
    func = toolbox.compile(expr=individual)
    return sum(func(*in_) == out for in_, out in zip(inputs, outputs)), individual.height

toolbox.register("evaluate", evalXor)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genGrow, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
    random.seed()

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats=mstats, halloffame=hof, verbose=True)

    # Print graph
    print(str(hof[0]))
    print(hof[0].height)
    print(hof[0].fitness.values)

    # Plot graph
    expr = hof[0]
    nodes, edges, labels = gp.graph(expr)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

    # Print log
    return pop, log, hof

if __name__ == "__main__":
    main()
