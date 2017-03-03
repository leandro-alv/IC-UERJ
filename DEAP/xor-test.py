#! ~/IC-UERJ/environment/python3

import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import matplotlib.pyplot as plt
import networkx as nx

pset = gp.PrimitiveSetTyped("MAIN", [bool,bool,bool,bool,bool], bool)
pset.addPrimitive(operator.xor, [bool, bool], bool)
pset.renameArguments(ARG0='a')
pset.renameArguments(ARG1='b')
pset.renameArguments(ARG2='c')
pset.renameArguments(ARG3='d')
pset.renameArguments(ARG4='e')

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):

    func = toolbox.compile(expr=individual)

    smpa = [bool(random.getrandbits(1)) for x in range(points)]
    smpb = [bool(random.getrandbits(1)) for x in range(points)]
    smpc = [bool(random.getrandbits(1)) for x in range(points)]
    smpd = [bool(random.getrandbits(1)) for x in range(points)]
    smpe = [bool(random.getrandbits(1)) for x in range(points)]

    errors = [func(a,b,c,d,e) == (operator.xor(e,operator.xor(d,operator.xor(c,operator.xor(a,b))))) for a,b,c,d,e in zip(smpa, smpb, smpc, smpd, smpe)]
    return errors.count(False)/points, individual.height

toolbox.register("evaluate", evalSymbReg, points=10)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
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

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    #print graph
    print(str(hof[0]))
    print(hof[0].height)

    #####
    #plot graph
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
    #####

    # print log
    return pop, log, hof

if __name__ == "__main__":
    main()
