import copy
import random
import sys
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators
from genetic import Population, format_trigger, get_expression, rand_trigger, evaluate, selection, crossover, mutation, get_indicator_and_candle_values_from_gene

MAX_ITER = 30
POPULATION = 501
USE_OPTUNA = len(sys.argv) == 2 and sys.argv[1] == 'Optuna'

# Seeding to ensure same initial conditions
SEED = random.randint(0, 100000)
def reset_seed():
	random.seed(SEED)
	np.random.seed(SEED)

# Used when USE_OPTUNA is False
DEFAULT_PARAMS = {
	'MUTATION_STD': 1.5, # The standard deviation of the normal distribution used to mutate
	'N_MUTATIONS': 15, # The number of genes to mutate
	'N_CROSSOVER': 250, # The number of genes to crossover
}

# Defines the searchspace for OPTUNA
OPTUNA_SEARCHSPACE = {
	'MUTATION_STD': list(np.linspace(0.1, 3, 20)),
	'N_MUTATIONS': list(range(5, 25)),
	'N_CROSSOVER': list(range(100, POPULATION, 25)),
}

def run(df_rows: list, indicators_and_candle_values, trial: optuna.trial=None):
	reset_seed()
	params = DEFAULT_PARAMS if trial is None else {
		'MUTATION_STD': trial.suggest_float('MUTATION_STD', OPTUNA_SEARCHSPACE['MUTATION_STD'][0], OPTUNA_SEARCHSPACE['MUTATION_STD'][-1]),
		'N_MUTATIONS': trial.suggest_int('N_MUTATIONS', OPTUNA_SEARCHSPACE['N_MUTATIONS'][0], OPTUNA_SEARCHSPACE['N_MUTATIONS'][-1]),
		'N_CROSSOVER': trial.suggest_int('N_CROSSOVER', OPTUNA_SEARCHSPACE['N_CROSSOVER'][0], OPTUNA_SEARCHSPACE['N_CROSSOVER'][-1]),
	}

	# Initialise gene pools
	pool: Population = [
		[
			rand_trigger(indicators_and_candle_values), # buy trigger
			rand_trigger(indicators_and_candle_values), # sell trigger
		]
		for _ in range(POPULATION)
	]

	next_gen: Population = copy.deepcopy(pool)

	# Record bot values for visualisation
	bot_record = []
	# Run genetic algorithm for some number of iterations
	epochs = range(MAX_ITER) if trial is not None else tqdm(range(MAX_ITER), total=MAX_ITER)
	for epoch in epochs:
		# Shuffle the pool to avoid bias
		random.shuffle(pool)
		max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)

		# Report to optuna
		if trial is not None:
			trial.report(fit_sum / len(pool), epoch)

			if epoch > 10 and trial.should_prune():
				raise optuna.TrialPruned()

		# Append the value record of the best bot
		bot_record.append({'max_pos': max_pos, 'max_fit': max_fit, 'fit_sum': fit_sum, 'fitnesses': fitnesses})

		# Preserve the best gene for the next generation
		next_gen[0] = copy.deepcopy(pool[max_pos])

		# Do crossover for the rest of genes
		for i in range(1, params['N_CROSSOVER'], 2):
			g1, g2 = [selection(fit_sum, fitnesses) for _ in range(2)]
			next_gen[i], next_gen[i + 1] = crossover(pool[g1], pool[g2])

		# Mutate a small number of the population randomly
		mutation(next_gen, n_mutations = params['N_MUTATIONS'], mutation_std = params['MUTATION_STD'])
		pool = copy.deepcopy(next_gen)

	# Print out the best gene after all the evolution
	max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)

	if trial is None:
		return (max_pos, max_fit, fit_sum, fitnesses), bot_record, pool
	return fit_sum / len(pool)

if __name__=='__main__':
	# Allow printing the entire data frame
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	# Set up data frame
	df = retrieve_data()
	indicators_and_candle_values = list_indicators_and_candle_values(df)
	df = add_all_indicators(df, indicators_and_candle_values)
	df.to_csv('data.csv', index=False)
	df_rows = [row for _, row in df.iterrows()]

	# Run the genetic algorithm with default values
	if not USE_OPTUNA:
		(max_pos, max_fit, fit_sum, fitnesses), bot_record, pool = run(df_rows, indicators_and_candle_values)
		expressions = [get_expression(expression, indicators_and_candle_values) for expression in get_indicator_and_candle_values_from_gene(pool[max_pos])]
		print(f'buy trigger: {format_trigger(expressions[:4])}')
		print(f'sell trigger: {format_trigger(expressions[4:])}')
		print(f'best bot earns ${max_fit:.5f}')
		pd.DataFrame(bot_record).to_csv('fitness.csv', index=True, index_label='epoch')

	# Do a hyperparameter search with Optuna
	else:
		study = optuna.create_study(direction="maximize",
									sampler = optuna.samplers.GridSampler(search_space=OPTUNA_SEARCHSPACE),
									pruner = optuna.pruners.MedianPruner(),
									storage = 'sqlite:///db.sqlite3',
									study_name = f'seed_{SEED:05}',
								   )
		study.optimize(lambda trial: run(df_rows, indicators_and_candle_values, trial=trial))
