### Speech Decoding Continuous Analysis
def create_time_locked_trials(phn_matrix, continuous_data, continuous_stim, lag_val = 100):
	start_frames = list(map(lambda ind: 
            np.floor((phn_matrix[ind,4]*1000)/5), 
            range(len(phn_matrix))))
    
    stop_frames = list(map(lambda ind: 
        np.ceil((phn_matrix[ind,5]*1000)/5), 
        range(len(phn_matrix))))
    
    phn_locked_trials = np.array(list(map(lambda ind: 
        continuous_data[(start_frames[ind] + lag_vect[n_time]).astype('int'):
        (stop_frames[ind] + lag_vect[n_time]).astype('int'),:], 
        range(len(start_frames)))))

    phn_start_times = phn_matrix[:,4]
    phn_stop_times = phn_matrix[:,5]

    stim_trials = np.array(list(map(lambda ind: 
        continuous_stim[np.argmin(abs(continuous_stim[:,0] - phn_start_times[ind])):
            np.argmin(abs(continuous_stim[:,0] - phn_stop_times[ind])), :], 
            range(len(phn_start_times)))))

    return phn_locked_trials, stim_trials



def logistic_regression_analysis()
	lag_vect = np.linspace(-300, 300, 50)/5
	r2_amp_time = np.zeros((50, 251))
	if feature_type == 'f1':
		f_ind = 1
	elif feature_type == 'f2':
		f_ind = 2
	else:
		f_ind = 3
	for n_time in range(50):    
		phn_locked_trials, stim_trials = create_time_locked_trials(phn_matrix, continuous_data, continuous_stim, lag_val = lag_vect[n_time])
	    mse = []
	    r2 = []
	    for ind in np.unique(label(phn_matrix[:,3])):
	        nd = StandardScaler().fit_transform(np.concatenate(phn_locked_trials[label(phn_matrix[:,3]) == ind]))
	        stim = np.concatenate(stim_trials[label(phn_matrix[:,3]) == ind])
	        stim = StandardScaler().fit_transform(stim.reshape(-1,1))
	        stim = stim[np.linspace(0,stim.shape[0]-1,nd.shape[0]).astype('int'),f_ind]
	        X_train, X_test, y_train, y_test = train_test_split(nd, stim, test_size=0.2, random_state=42)
	        huber = LinearRegression().fit(X_train, y_train.reshape(-1,1))
	        stim_pred = huber.predict(X_test)
	        r2.append(r2_score(y_test, stim_pred))
	    r2 = np.array(r2)
	    r2_amp_time[n_time, :] = r2
	
	return r2
    
    
