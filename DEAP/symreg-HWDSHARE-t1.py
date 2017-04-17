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

#####################
#Predefined test functions
def f1(a, b, c):
    return ((a*b)+c)

def f2(a, b):
    return (a+b)
#####################

#####################
#Conbinatotion of possible entries in the selectors (n * 2**n) -> 2**n
def sel_conb(n):
    N_COMB = 2**n
    comb = [[False] * n for i in range(N_COMB)]
    for i in range(N_COMB):
        value = i
        divisor = N_COMB
        for j in range(n):
            divisor /= 2
            if value >= divisor:
                comb[i][j] = True
                value -= divisor
            else:
                comb[i][j] = False

    return comb
#####################

def if_then_else(condition, out1, out2):
    return out1 if condition else out2

def pass_bool(in1):
    return in1

pset = gp.PrimitiveSetTyped("MAIN", [float, float, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(pass_bool, [bool], bool) #We need to have one primitive with pass a bool
pset.renameArguments(ARG0='a')
pset.renameArguments(ARG1='b')
pset.renameArguments(ARG2='c')
pset.addTerminal(bool, bool, name="sel")

creator.create("FitnessMaxMin", base.Fitness, weights=(2.0, -1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMaxMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, smpa, smpb, smpc):

    index = 0
    selectors = []
    for node in individual: # > (2**individual.height) + 1
        if node.name == "sel":
            selectors.append(index)

        index += 1

    n_sel = len(selectors)
    comb = sel_conb(n_sel) # n_sel * 2**n_sel
    n_comb = len(comb)
    total_hits = 0 if n_sel >= 1 else -1
    raif1, raif2 = 0, 0

    for i in range(n_comb): # n_comb * n_sel
        for j in range(n_sel):
            individual[selectors[j]].value = comb[i][j]
        func = toolbox.compile(expr=individual)
        #Testing the expression with the predefined functions
        for a,b,c in zip(smpa,smpb,smpc):
            result = func(a,b,c)
            if result == f1(a,b,c):
                raif1 += 1
            elif result == f2(a,c):
                raif2 += 1
            else:
                total_hits -= 1
    total_hits += (raif1 + raif2)
    total_hits = 4 * total_hits if raif1 == raif2 else total_hits
        
    return total_hits/n_comb, n_sel, individual.height

smpa = random.sample([float(x) for x in range(-10,10)], k=5)
smpb = random.sample([float(x) for x in range(-10,10)], k=5)
smpc = random.sample([float(x) for x in range(-10,10)], k=5)

toolbox.register("evaluate", evalSymbReg, smpa=smpa, smpb=smpb, smpc=smpc)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
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
    print(hof[0].fitness.values)

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
