import pandas as pd
from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy
import warnings

# df = pd.read_csv('AS14.01.csv')

# t = len(df["mood"])-2
# training = df["mood"][:t]
# test = df["mood"][t]
# print("test ", test)


# model = ARIMA(training.tolist(), order=(0,1,1))
# model_fit = model.fit(disp=0)

# # one-step out-of sample forecast
# forecast = model_fit.forecast()[0]
# print("Forecast: {} Actual: {}".format(forecast[0], test))

df = pd.read_csv('data_mounir/AS14.03.csv')
dataset = df["mood"].tolist()

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])

	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error, predictions


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse,_ = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


def plot(predictions):
	df["mood"].plot()
	tmp = [None] * (len(df["mood"]) - len(predictions))
	tmp.extend(predictions)
	pyplot.plot(tmp)
	pyplot.show()
	

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
# evaluate_models(dataset, p_values, d_values, q_values)
_,predictions = evaluate_arima_model(dataset, (1,0,1))
plot(predictions)