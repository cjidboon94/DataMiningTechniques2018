import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def get_predictions(X, test, arima_order):
	model = ARIMA(X, order=arima_order)
	model_fit = model.fit(disp=0)
	return model_fit.forecast(steps=len(test))[0]


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, test, arima_order):
	predictions = get_predictions(X, test, arima_order)

	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, test, p_values, d_values, q_values):
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, test, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
	return best_cfg


def train_model(dataset, test_set):
	p_values = [0, 1, 2, 4]
	d_values = range(0, 3)
	q_values = range(0, 3)
	return evaluate_models(dataset, test_set, p_values, d_values, q_values)


def evaluate_model(train, test, best_cfg):
	predictions = get_predictions(train, test, best_cfg)
	errors = np.subtract(test, predictions)

	return errors


def dump(errors, dump_file):
	thefile = open(dump_file, 'w+')
	for item in errors:
		thefile.write("%s\n" % item)


best_cfg = ()
dirs = ["fold_1", "fold_2", "fold_3", "fold_4"]
for dirr in dirs:
	user = "AS14.01.csv"
	train_file = "data_normalized/{}/train/{}".format(dirr, user)
	test_file = "data_normalized/{}/test/{}".format(dirr, user)
	dump_file = "data_normalized_res/arima/{}/{}".format(dirr, user)

	df = pd.read_csv(train_file)
	moods_train = df["mood"].tolist()
	df = pd.read_csv(test_file)
	moods_test = df["mood"].tolist()

	if (dirr == dirs[0]):
		best_cfg = train_model(moods_train, moods_test)

	errors = evaluate_model(moods_train, moods_test, best_cfg)
	dump(errors, dump_file)