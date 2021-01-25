import json
import os
import pyriemann as pr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from mne.decoding import Vectorizer, CSP
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from pyriemann.classification import MDM
import numpy as np
import mne
import scipy.io as sio

def save_to_VH(filename, varname ='outputMetric', value=np.nan):
    output_path = os.getenv('VH_OUTPUTS_DIR')
    filepath = os.path.join(output_path, filename)
    save_dict = {varname[var_ind] : value[var_ind] for var_ind in range(len(value))}
    sio.savemat(filepath + '.mat', save_dict)

def create_valohai_meta(varname, value):
    meta_dict = {varname[var_ind] : value[var_ind] for var_ind in range(len(varname))}
    print(json.dumps(meta_dict))

def moses_chang_pipeline(reduced_puretone_epochs.copy(), sdr_params_, fname):
	if hilbertFlag:
	    X = np.abs(epochs.apply_hilbert().get_data())
	else:
	    X = epochs.get_data()

	y = epochs.events[:,2]

	print(X.shape)

	from sklearn.decomposition import PCA
	pca = PCA(n_components=150)

	X = pca.fit_transform(X.reshape(X.shape[0], -1))

	X_train, X_test, y_train, y_test = train_test_split(X, y,\
	    test_size= params['test_size_'], random_state=13, stratify = y)

	clf = LinearDiscriminantAnalysis(
                    solver = params['solver_'], 
                    shrinkage = params['clf_reg_'])

	fitted_clf = clf.fit(X_train, y_train)
    preds_test = clf.predict(X_test)

    # Aggressive epoch rejection can wipe out entire classes making the target names different
    # Handle here:

    ind_ep = [ ep in np.unique(epochs.events[:,2]) for ep in np.arange((len(epochs.event_id.keys())))+1 ]
    trg_nms_ = [nm for nm,i_ in zip(list(epochs.event_id.keys()),ind_ep) if i_]
    report = classification_report(y_test, preds_test,\
        target_names = trg_nms_)

    fig0 = plt.figure()
    cm = confusion_matrix(y_test, preds_test)
    print(np.unique(y_test, return_counts = True)[1][0]*100/len(y_test))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm*100);plt.colorbar();
    plt.ylabel('True label');plt.xlabel('Predicted label')
    plt.title('LDA' + ' Accuracy Score:' + str((accuracy_score(y_test, preds_test)*100).round(3)) + '% (Chance: ' + str(np.array(np.unique(y_test, return_counts = True)[1][0]*100/len(y_test)).round(2)) + '%)')
    save_image_to_VH(fig0, 'dc_results_' + str(nChannels_))
    f1_scores = np.mean(f1_score(y_test, preds_test,\
    	average = 'micro'))

    return f1_scores

def supervised_dimensional_reduction(epochs, method, params, fname,\
    hilbertFlag = True, saveFlag = True, reconFlag = True):
    
    """
    Allows for supervised dimensional reductions 
    Inputs:
        Epochs = Packed 
        Method = Method List of pipelined transforms
            eg. ['standardscaler', 'csp', 'log']
            or ['vectorizer', 'xdawn', 'minmax', 'lda']
        
        Params = These are parameters that are unpacked 
        
        **** How to build params *********
        At the top of your notebook build your params by
        placing this line after the parameters box with all
        relevant parameters in this format "var_"
        
        params = {item : eval(item) for item in dir() if not item.startswith("_") and item.endswith('_')}
        
        to fill in variables in specific methods
                eg. {n_comps : 16, reg: 0.1, penalty:'l1/l2',
                l1_ratio: 0.3}

        **** How to build params *********

    Outputs:
        Dimensionally Reduced Epochs 
    """
    ## Split up data into hold out set for proper cross validation 

    if hilbertFlag:
        X = np.abs(epochs.apply_hilbert().get_data())
    else:
        X = epochs.get_data()
    
    y = epochs.events[:,2]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,\
        test_size= params['test_size_'], random_state=13)

    cv = StratifiedKFold(n_splits=params['n_splits_'], shuffle=True)

    # Create Pipeline
    pipe_var = []
    print(method)


    for method_var in method:

        'Running {dim_red_method}...'.format(dim_red_method = method_var)

        ## Dimensional Reductions
        if method_var.lower() == 'xdawn':
            pipe_var.append(('xdawn',  pr.estimation.XdawnCovariances(params['n_components_'], estimator = params['estimator_'], xdawn_estimator = params['estimator_'])))
            vectorizerFlag = True
        
        elif method_var.lower() == 'erpcov':
            pipe_var.append(('ERPCovariances',   pr.estimation.ERPCovariances(estimator = params['estimator_'], svd = params['n_components_']))) # svd here should be e.g. 4. Its not akin to rank of outputspace wrt stimuli
            vectorizerFlag = True

        elif method_var.lower() == 'hankelcov':
            pipe_var.append(('HankelCovariances',   pr.estimation.HankelCovariances(estimator = params['estimator_'], delays = [0, 1, 8, 12, 64])))
            vectorizerFlag = True

        elif method_var.lower() == 'tangent_space':
            pipe_var.append(('TS',     pr.tangentspace.TangentSpace(metric = 'riemann')))
            
        elif method_var.lower() == 'csp':
            pipe_var.append(('csp',    CSP(n_components = params['n_components_'], reg = params['sdr_reg_'])))

        elif method_var.lower() == 'pr_csp':
            pipe_var.append(('pr_csp', pr.spatialfilters.CSP(params['n_components_'], metric = 'euclid', log = False))) # less control here over reg and could use riemann distance instead?


        ## Scalers
        elif method_var.lower() == 'standard':
            pipe_var.append((method_var, StandardScaler()))

        elif method_var.lower() == 'minmax':
            pipe_var.append((method_var, MinMaxScaler()))

        elif method_var.lower() == 'vectorizer':
            pipe_var.append((method_var, Vectorizer()))

        ## Models 
        elif method_var.lower() == 'lda':                 
            pipe_var.append((method_var,
                LinearDiscriminantAnalysis(
                    solver = params['solver_'], 
                    shrinkage = params['clf_reg_'])))


        elif method_var.lower() == 'logistic':
            if vectorizerFlag:
                pipe_var.append(('Vectorizer', Vectorizer()))

            pipe_var.append((method_var,
                LogisticRegression(
                    solver = params['solver_'],
                    C = params['clf_reg_'], # add in regularisation (small is more reggie)
                    n_jobs = -1,
                    penalty = params['penalty_'])))

        elif method_var.lower() == 'mdm':
            pipe_var.append(('MDM', pr.classification.MDM(metric = 'riemann', n_jobs = -2)))
        
        elif method_var.lower() == 'fgmdm':
            pipe_var.append(('FgMDM', pr.classification.FgMDM(metric = 'riemann', n_jobs = -2)))
            
        elif method_var.lower() == 'None':
            continue
            
    print(pipe_var)
    ## Actual Pipeline Run 
    clf = Pipeline(pipe_var)
    
    #scores = cross_val_score(clf, X_train, y_train,\
    #        cv = cv, n_jobs = -1, pre_dispatch = 12)
    #    create_valohai_meta(['validation_accuracy'], [np.mean(scores)])

    ## Predict on left out data
    fitted_clf = clf.fit(X_train, y_train)

    preds_test = clf.predict(X_test)

    # Aggressive epoch rejection can wipe out entire classes making the target names different
    # Handle here:
    ind_ep = [ ep in np.unique(epochs.events[:,2]) for ep in np.arange((len(epochs.event_id.keys())))+1 ]
    trg_nms_ = [nm for nm,i_ in zip(list(epochs.event_id.keys()),ind_ep) if i_]
    report = classification_report(y_test, preds_test,\
        target_names = trg_nms_)

    print(report)

    f1_scores = np.mean(f1_score(y_test, preds_test,\
        average = 'micro'))

    ## TODO MAKE THIS A FLAG FOR OFFLINE TESTING
    create_valohai_meta(['test_accuracy'], [f1_scores])

    ## Transform epochs to reduced format
    X_dim_red = fitted_clf.steps[0][1].transform(X)

    ## Saving 
    if saveFlag:
        if 'csp' in method:
            save_to_VH(fname+'CSP',\
                ['patterns', 'csp_epochs', 'test_predictions', 'test_labels'],\
                [fitted_clf.steps[0][1].patterns_, X_dim_red, preds_test, y_test])
        elif 'xdawn' in method:
            try:
                save_to_VH(fname+'XDawn',\
                    ['patterns', 'test_predictions', 'test_labels'],\
                    [fitted_clf.steps[0][1].Xd_.patterns_.T, preds_test, y_test])
            except:
                save_to_VH(fname+'XDawn',\
                    ['test_predictions', 'test_labels'],\
                    [preds_test, y_test])
        else:
            save_to_VH(fname+ method[0],\
                    ['test_predictions', 'test_labels'],\
                    [preds_test, y_test])
                
    if reconFlag:
        if 'csp' in method:
            filters = fitted_clf.steps[0][1].filters_
        elif 'xdawn' in method:
            filters = fitted_clf.steps[0][1].Xd_.filters_
            print(filters.shape)
            print(X.shape)
            if filters.shape[0] != X.shape[1]:
                filters = filters.T
            
        recon_epochs = np.asarray([np.dot(epoch.T, filters) for epoch in epochs])
        recon_epochs = np.asarray([np.dot(epoch, filters.T) for epoch in recon_epochs])
        recon_epochs = np.asarray([epoch.T for epoch in recon_epochs])
        revised_epochs = mne.EpochsArray(recon_epochs, epochs.info, epochs.events, epochs.tmin, epochs.event_id)
        return revised_epochs
    else:
        return X_dim_red


def hier_binning(labels, split_int = 3):

    bins = np.linspace(1,max(labels),split_int)
    bins = np.delete(bins, -1)
    ind = np.digitize(labels, bins)

    return ind


def parameter_tuning(epochs, pipeline, params, test_size_ = 0.2):
    """
    Allow for double nested cross validation for 
    hyperparameter search in grid-based way - upgrading to hyperopt soon
    Allows for train, test and validation sets to create best hyperparameters

    Inputs:
        epochs
        pipeline to search hyperparameters in 
        parameters to search through 
    Outputs:
        best predictions
        F1 score
        best parameters 
        CSP object
    """
    X = epochs
    y = epochs.events[:,2]
    # cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size_, random_state=13)

    grid_search = GridSearchCV( pipeline,
        cv = 5,
        n_jobs = -1,
        pre_dispatch = 'n_jobs/4',
        param_grid = params,
        scoring = 'roc_auc',
        refit = 'AUC')
    
    grid_search.fit(X_train, y_train)
    
    preds = grid_search.predict(X_test)

    print("Best score: %0.3f" % grid_search.best_score_)
    
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    decomp_sup_best = pipeline.step[0][0]
    epochs = decomp_sup_best.transform(X)
    # epochs = map.pool([X_train, X_test, best_parameters[param_name]], FitCSPData)

    return preds, best_parameters, epochs, decomp_sup_best
