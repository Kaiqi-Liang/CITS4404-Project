import pandas as pd
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators
from genetic import POPULATION, Population, rand_trigger, evaluate, selection, crossover, mutation, get_indicator_and_candle_values_from_gene

MAX_ITER = 30

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = retrieve_data()
indicators_and_candle_values = list_indicators_and_candle_values(df)
df = add_all_indicators(df, indicators_and_candle_values)
pool: Population = [
    [
		*rand_trigger(indicators_and_candle_values),
		*rand_trigger(indicators_and_candle_values),
		*rand_trigger(indicators_and_candle_values),
		*rand_trigger(indicators_and_candle_values),
	]
    for _ in range(POPULATION)
]
next: Population = [[] for _ in range(POPULATION)]

for iter in range(MAX_ITER):
	print(iter)
	max_pos, max_fit, fit_sum, fitnesses = evaluate(df, pool, indicators_and_candle_values)
	# Preserve best gene and replicate it so it appears twice in the next generation
	next[0] = pool[max_pos]
	next[1] = pool[max_pos]
	for i in range(2, POPULATION - 1, 2):
		crossover(pool[selection(fit_sum, fitnesses)], pool[selection(fit_sum, fitnesses)], i, next)
	mutation(next, indicators_and_candle_values)
	pool = next

max_pos, max_fit, fit_sum, fitnesses = evaluate(df, pool, indicators_and_candle_values)
print(max_fit)
a, b, c, d, e, f, g, h, i, j, k, l = get_indicator_and_candle_values_from_gene(pool[max_pos], indicators_and_candle_values)
print(f'buy trigger: {a} > {b} * {c} & {d} > {e} * {f}')
print(f'sell trigger: {g} > {h} * {i} & {j} > {k} * {l}')
