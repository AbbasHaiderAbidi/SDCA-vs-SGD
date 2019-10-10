import numpy as np
import random as rnd
import time as tm

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

eta=1/25000

def gradFunc( theta, y, X, C ):
    w = theta[0:]
    n = y.size
    i = rnd.randint( 0, n-1 )
    x = X[i,:]
    discriminant = (x.dot( w )) * y[i]
    g = 0
    if discriminant < 1:
        g = -1
    delw = w +2*n* C  * (1 - np.multiply( (x.T.dot( w )), y[i] )) * (x * g) * y[i]
    return delw 

def stepFunc( eta, t ):
    return eta/t


def doGD( theta, y, X, C, t):
    delta = gradFunc( theta , y, X, C)
    theta = theta - stepFunc(eta,t+1 ) * delta
    return theta

################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc
	theta_SGD = w
	cumulative = w

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
		theta_SGD = doGD(theta_SGD, y, X, C, t)
		cumulative = cumulative + theta_SGD
		w = cumulative/(t+1)
		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		
		# Please note that once timeout is reached, the code will simply return w, b
		# Thus, if you wish to return the average model (as we did for GD), you need to
		# make sure that w, b store the averages at all times
		# One way to do so is to define two new "running" variables w_run and b_run
		# Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w and b
		# w = (w * (t-1) + w_run)/t
		# b = (b * (t-1) + b_run)/t
		# This way, w and b will always store the average and can be returned at any time
		# w, b play the role of the "cumulative" variable in the lecture notebook
		# w_run, b_run play the role of the "theta" variable in the lecture notebook
		
	return (w, b, totTime) # This return statement will never be reached