import numpy as np

def pop_years(dirs, years):
    i = 0
    while i < len(dirs):
        dir = dirs[i]
        for year in years:
            if year in dir:
                dirs.pop(i)
                i -= 1
                break
        i += 1

def r2(y_true, y_pred):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    sse  = np.square( y_pred - y_true ).sum()
    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - sse/sst

def soundingNum(soundingID):
	id = np.double(soundingID)
	return np.floor((id+0.01)%10)