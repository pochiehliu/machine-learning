import numpy as np
import pandas as pd
from numpy.linalg import svd

def get_data():
	# read data using pandas and conver to np array
	x_train = pd.read_csv('X_train.csv').values
	y_train = pd.read_csv('y_train.csv').values
	x_test = pd.read_csv('X_test.csv').values
	y_test = pd.read_csv('y_test.csv').values
	return x_train, y_train, x_test, y_test

def normalize(x,y, x_t = None, y_t = None):
	# substract mean and devide by variance
	if x_t is None:
		nor_x = (x[:,0:6] - np.mean(x[:,0:6], axis = 0)) / np.sqrt(np.var(x[:,0:6], axis = 0))
		nor_x = np.hstack( (nor_x, x[:,6:7]) )
		# substract mean for y
		nor_y = y - np.mean(y)
	else:
		# use training data mean and std to normalize test data
		nor_x = (x_t[:,0:6] - np.mean(x[:,0:6], axis = 0)) / np.sqrt(np.var(x[:,0:6], axis = 0))
		nor_x = np.hstack( (nor_x, x_t[:,6:7]) )
		nor_y = y_t - np.mean(y)
	return nor_x, nor_y

def w_solver(x,y, lambd):
	# apply svd to solve ridge regression
	u, s, vh = svd(x, full_matrices = False)
	# solve df = trace s**2/(s**2+lambd)
	# solve w_rr using formula w_RR = V * S_lambda^-1 * U.T * y
	return  np.sum(s**2/(s**2+lambd)), vh.T.dot(np.diag(s/(s**2+lambd))).dot(u.T).dot(y)[:,0]

def transform_df(w):
	# add feature names
	df = pd.DataFrame(w)
	df.columns = ['df','cylinders','displacement',
					'horsepower','weight','acceleration',
					'year made','intercept']
	return df

def plot_3a(df,title):
	ax = df.plot(x='df', figsize= (8,6))
	_ = ax.set_xlabel('df(lambda)')
	_ = ax.set_ylabel('Ridge Regression Coefficient W')
	_ = ax.set_title('Problem 3 (a)')
	ax.get_figure().savefig(title)

def problem_3a(x, y):
	# weights' log
	w = np.zeros([5001,8])
	
	for lambd in range(0,5001):
		w[lambd,0], w[lambd,1:] = w_solver(x,y,lambd)
	
	w_df = transform_df(w)
	plot_3a(w_df,'a_normalized.png')

def rmse(y,y_h):
    return np.sqrt( np.sum((y-y_h)**2)/len(y) )

def transform_rmse(result):
	df = pd.DataFrame(result)
	df.columns = ['lambda','RMSE']
	return df

def plot_3c(df, title):
	ax = df.plot(x='lambda', figsize= (8,6))
	_ = ax.set_xlabel('lambda')
	_ = ax.set_ylabel('RMSE of test data set')
	_ = ax.set_title('Problem 3 (c)')
	ax.get_figure().savefig(title)

def problem_3c(x, y, x_t, y_t):
	result = np.zeros([51,2])

	for lambd in range(0,51):
		_, w_RR = w_solver(x,y,lambd)
		y_h = x_t.dot(w_RR)		
		result[lambd,0], result[lambd,1] = lambd, rmse(y_t,y_h)
		
	df = transform_rmse(result)
	plot_3c(df,'c_normalized.png')

def p_th_transform(x, p=1, x_t =None):
	if x_t is None:
		temp = x[:,0:6]**p
		return (temp-np.mean(temp,axis = 0)) / np.sqrt(np.var(temp,axis = 0))
	else:
		temp1 = x[:,0:6]**p
		temp2 = x_t[:,0:6]**p
		return (temp2-np.mean(temp1,axis = 0)) / np.sqrt(np.var(temp1,axis = 0))

def problem_3d(x, y, x_t, y_t, raw_x, raw_x_t):
	# generate p = 1, 2, 3 of x
	p1_x = x
	p2_x = np.hstack((p_th_transform(raw_x,2), p1_x))
	p3_x = np.hstack((p_th_transform(raw_x,3), p2_x))
	p1_x_t = x_t
	p2_x_t = np.hstack((p_th_transform(raw_x, 2, raw_x_t), p1_x_t))
	p3_x_t = np.hstack((p_th_transform(raw_x, 3, raw_x_t), p2_x_t))

	# solve d
	result = np.zeros([101,4])
	for lambd in range(0,101):
		_, w_RR1 = w_solver(p1_x,y,lambd)
		_, w_RR2 = w_solver(p2_x,y,lambd)
		_, w_RR3 = w_solver(p3_x,y,lambd)
		y_h1, y_h2, y_h3 = p1_x_t.dot(w_RR1), p2_x_t.dot(w_RR2), p3_x_t.dot(w_RR3)
		result[lambd,:] = (lambd, rmse(y_t,y_h1), rmse(y_t,y_h2), rmse(y_t,y_h3))

	df = pd.DataFrame(result)
	df.columns = ['lambda','RMSE (p=1)', 'RMSE (p=2)', 'RMSE (p=3)']

	ax = df.plot(x='lambda', figsize= (8,6))
	_ = ax.set_xlabel('lambda')
	_ = ax.set_ylabel('RMSE of test data set')
	_ = ax.set_title('Problem 3 (d)')
	ax.get_figure().savefig('d_normalized')

def main():
	# import data
	x_train, y_train, x_test, y_test = get_data()
	# normalize training data
	nor_x, nor_y = normalize(x_train, y_train)
	# normalize testing data
	nor_x_t , nor_y_t= normalize(x_train, y_train, x_test, y_test)
	# solve problem 3a
	problem_3a(nor_x, nor_y)
	print('Problem 3 (a) done')
	# solve problem 3c
	problem_3c(nor_x, nor_y, nor_x_t, nor_y_t)
	print('Problem 3 (c) done')
	# solve problem 3d
	problem_3d(nor_x, nor_y, nor_x_t, nor_y_t, x_train, x_test)
	print('Problem 3 (d) done')

if __name__ == "__main__":
	main()