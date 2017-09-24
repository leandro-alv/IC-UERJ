import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#####################
#Predefined functions
def f1(a, b, c):
    return ((a*b)+c)

def f2(a, b):
    return (a+b)
#####################

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


def if_then_else(condition, out1, out2):
    return out1 if condition else out2

def pass_bool(in1):
	return in1

pset = gp.PrimitiveSetTyped("MAIN", [float, float, float], float)
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(if_then_else, [bool, float, float], float)
pset.addPrimitive(pass_bool, [bool], bool)
pset.renameArguments(ARG0='a')
pset.renameArguments(ARG1='b')
pset.renameArguments(ARG2='c')
#pset.renameArguments(ARG3='d')
pset.addTerminal(bool, bool, name="sel")
#pset.addEphemeralConstant("sel", lambda: random.choice((True,False)), bool)

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def main():

	#print(pset.terminals)
	#print(pset.primitives)
	
	ind = toolbox.individual()
	i = 0
	if_then_elses = []
	for node in ind:
		if node.name == "if_then_else":
			if_then_elses.append(i)

		i += 1
	'''
	i = 0
	selectors = []
	for node in ind:
		if node.name == "sel":
			selectors.append(i)

		i += 1
	n_sel = len(selectors)
	comb = sel_conb(n_sel)
	n_comb = len(comb)

	for i in range(n_comb):
		for j in range(n_sel):
			ind[selectors[j]].value = comb[i][j]
			print(ind)
			func = toolbox.compile(expr=ind)
			print(func(1,2,3))

	print("Selectors:")
	for sel in selectors:
		print("{} {} {}".format(sel, ind[sel].name, ind[sel].value))
	
	print("{}".format(comb))
	'''
	print(ind)
	for ifs in if_then_elses:
		subTree = ind.searchSubtree(ifs)
		print("{}".format(subTree))
		string = ""
		for n in range(subTree.start, subTree.stop):
			string += ind[n].name + " "
		print(string)

	return 0

if __name__ == "__main__":
    main()