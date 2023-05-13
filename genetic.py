import copy
import random
import numpy as np
import pandas as pd
from data import StrSeriesPairs

# Language to express candidate solutions which are defined by Gene, a subset of the dnf
Expression = tuple[str, float, str]
Literal = tuple[int, float, int]
Conjunctive = tuple[Literal, Literal]
Disjunctive = tuple[Conjunctive, Conjunctive]
Gene = tuple[Disjunctive, Disjunctive]
Population = list[Gene]

def rand_trigger(indicators_and_candle_values: StrSeriesPairs) -> Disjunctive:
    return [
		[
			rand_expression(indicators_and_candle_values),
			rand_expression(indicators_and_candle_values),
		],
		[
			rand_expression(indicators_and_candle_values),
			rand_expression(indicators_and_candle_values),
		],
	]

def rand_expression(indicators_and_candle_values: StrSeriesPairs) -> Literal:
	'''
	Randomly create an expression of `A > c * B` where `A` and `B` are indicator or candle values.
	'''
	lhs, rhs = random.sample(range(len(indicators_and_candle_values)), 2)
	c = np.random.normal(0, 1, 1)[0]

	return [lhs, c, rhs]

def evaluate(df_rows: pd.Series, pool: Population, indicators_and_candle_values: StrSeriesPairs):
	'''
	Call the fitness evaluation for each gene in the pool and return the restuls in fitnesses.
	Add up the total fitness sum of all the genes in the whole generation and find the best one to return.
	'''
	fitnesses = [fitness(df_rows, gene, indicators_and_candle_values) for gene in pool]
	fit_sum = sum(fitnesses)
	max_pos = np.argmax(fitnesses)
	max_fit = fitnesses[max_pos]
	return max_pos, max_fit, fit_sum, fitnesses

def get_indicator_and_candle_values_from_gene(gene: Gene) -> list[Literal]:
	'''
	Return the buy and sell triggers in the expression of:
	`buy_trigger = A and B or C and D`,
	`sell_trigger = E and F or G and H`,
	where A, B, C, D, E, F, G, H are expressions in the form of `a > c * b`.
	'''
	buy_trigger, sell_trigger = gene
	buy_conj1, buy_conj2 = buy_trigger
	sell_conj1, sell_conj2 = sell_trigger
	A, B = buy_conj1
	C, D = buy_conj2
	E, F = sell_conj1
	G, H = sell_conj2
	return [A, B, C, D, E, F, G, H]

def get_expression(expression: Literal, indicators_and_candle_values: StrSeriesPairs) -> Expression:
	a, b, c = expression
	return indicators_and_candle_values[a][0], b, indicators_and_candle_values[c][0]

def evaluate_trigger(row, trigger: list[Expression]):
	'''
	Evaluate a trigger, return False if any of the values is `nan`.
	'''
	return all(map(lambda exp: row[exp[0]] > exp[1] * row[exp[2]], trigger))

def fitness(df_rows: pd.Series, gene: Gene, indicators_and_candle_values: StrSeriesPairs) -> float:
	'''
	Calculate the amount of money at the end of all the trades starting with $100.
	'''
	amount = 100.
	buy_trigger = False
	expressions = [get_expression(expression, indicators_and_candle_values) for expression in get_indicator_and_candle_values_from_gene(gene)]
	for row in df_rows:
		if not buy_trigger:
			if evaluate_trigger(row, expressions[:4]):
				amount -= amount * 0.02
				amount /= row['close']
				buy_trigger = True
		else:
			if evaluate_trigger(row, expressions[4:]):
				amount *= row['close'] * (1 - 0.02)
				buy_trigger = False
	# Force sell at the end
	if buy_trigger:
		amount *= row['close'] * (1 - 0.02)

	return amount

def selection(fit_sum: float, fitnesses: list[float]):
	'''
	Implement wheel of fitness to select a random gene biased to reflect its fitness.
	Wheel segments are covered with gene symbols, where the segment size matches the relative fitness level (twice the fitness means twice the area).
	Assuming each spin is random, it will select a random gene with the desired bias according to their fitness level.
	'''
	return np.random.choice(range(len(fitnesses)), p=[fitness / fit_sum for fitness in fitnesses])

def crossover(g1: Gene, g2: Gene):
	'''
	Given `2` genes `g1` and `g2` which were selected through the selection process.
	Randomly pick a cut out of the `4` expressions.
	Then the left side of the cut of `g1` is glued to the right side of the cut of `g2` and vice versa.
	These two new genes will enter the next generation of the process.
	`buy_trigger = A and B or C and D`,
	`sell_trigger = E and F or G and H`,
 	'''
	expressions1 = get_indicator_and_candle_values_from_gene(g1)
	expressions2 = get_indicator_and_candle_values_from_gene(g2)
	cut = random.choice(range(1, 7))
	A, B, C, D, E, F, G, H = expressions1[:cut] + expressions2[cut:]
	I, J, K, L, M, N, O, P = expressions2[:cut] + expressions1[cut:]
	return copy.deepcopy([[[A, B], [C, D]], [[E, F], [G, H]]]), copy.deepcopy([[[I, J], [K, L]], [[M, N], [O, P]]])

def mutation(pool: Population, n_mutations = 5, mutation_std = 1):
	'''
	Randomly change the constant value in one of the 8 expressions in a small percentage of the genes picked randomly from the population.
 	'''
	for _ in range(n_mutations):
		# Skip the first one so that the best gene will never be mutated
		i = random.randrange(1, len(pool))
		a,b,c = [random.choice([0,1]) for _ in range(3)]
		expression = pool[i][a][b][c]

		expression[1] = np.random.normal(0, mutation_std, 1)[0]

def format_trigger(expressions: list[Expression]):
	formatted = [f'{exp[0]} > {exp[1]:.5f} * {exp[2]}' for exp in expressions]
	return '( {} && {} ) || ( {} && {} )'.format(*formatted)
