import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def drop_na_columns(dataframe, list_of_columns, threshold):
    """Drop columns where number of null entries in a column exceeds a user-set percentage threshold"""
    n = dataframe.shape[0]
    to_drop = [column for column in list_of_columns if (dataframe[column].isnull().sum() / n) > threshold]
    dataframe.drop(to_drop, axis = 1, inplace = True)
    print ('Number of dropped columns: {}'.format(len(to_drop)))
    print ('\n')
    print ('Dropped columns: \n', to_drop)
    
def categorical_and_discrete_na_filler(dataframe, categorical_columns):
    """Fill empty rows with values from selected column according to current distribution percentages"""
    for column in categorical_columns:
        choice = sorted(dataframe[dataframe[column].notnull()][column].unique())
        probability = dataframe[column].value_counts(normalize = True).sort_index().values
        dataframe[column] = dataframe[column].apply(
            lambda x: np.random.choice(choice, p = probability) 
            if (pd.isnull(x)) 
            else x)
        
def continuous_na_filler(dataframe, columns, method):
    """Fill empty rows with values according to user-chosen method; mean or median"""
    if method == 'mean':
        for column in columns:
            value = np.mean(dataframe[column])
            dataframe[column].fillna(round(value, 0), inplace = True)
    elif method == 'median':
        for column in columns:
            value = np.nanmedian(dataframe[column])
            dataframe[column].fillna(round(value, 0), inplace = True)
    else:
        print ('Method not available. Please choose either mean or median, else update function for desired method.')
        
def check_outliers(dataframe, list_of_columns, lower_quantile_list, upper_quantile_list):
    """Returns a dataframe of outliers according to user provided quantiles"""
    quantile = lower_quantile_list + upper_quantile_list

    summary_dict = {}
    for col in list_of_columns:
        summary_dict[col] = []
        for i in quantile:
            summary_dict[col].append(dataframe[col].quantile(i))

    summary_df = pd.DataFrame(summary_dict)
    summary_df_final = pd.concat([pd.DataFrame(quantile, columns=['Quantile']), summary_df], axis = 1)

    return summary_df_final

def drop_values_multi(dataframe, list_of_columns, quantile):
    """Drop outliers based on quantile """
    to_drop_index = []
    quantile = quantile

    for i in list_of_columns:
        index = list(dataframe[dataframe[i] > dataframe[i].quantile(quantile)].index)
        to_drop_index = to_drop_index + index

    dataframe.drop(set(to_drop_index), axis = 0, inplace = True)
    print ('Successfully dropped rows!')
    
    
def central_limit_mean(dataset, sample_size = 50, num_simulations = 500, return_mean = False):    
    """TBD"""
    random_chosen = [np.mean(np.random.choice(dataset, size = sample_size)) for i in range(num_simulations)]
    if return_mean == False:
        return random_chosen
    else:
        return (random_chosen, round(np.mean(random_chosen), 2))
    
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    import statsmodels.api as sm
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

def opt_plots(opt_model):
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    plt.figure(figsize=[8,5])
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    
def make_radar_chart(data, title, figsize):
    import math

    categories = list(data.columns)
    N = len(categories)
    plt.figure(figsize = figsize)
    ax = plt.subplot(111, polar = True)
    ax.set_theta_offset(math.pi/2)
    ax.set_theta_direction(-1)

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    plt.xticks(angles[:-1], categories)

    ax.set_rlabel_position(30)
    maximum = data.max().sort_values()[-1]
    plt.yticks(np.linspace(0, maximum + 0.1, 10), color = 'red', size = 8)
    plt.ylim(0, maximum + 0.1)

    values = data.iloc[0,:].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth = 1, linestyle = 'solid', label = data.index[0], color = 'black')
    ax.fill(angles, values, 'b', alpha = 0.1)

    values = data.iloc[1,:].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth = 1, linestyle = 'solid', label = data.index[1], color = 'red')
    ax.fill(angles, values, 'o', alpha = 0.1)

    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')

    plt.legend(labels = ['Non-Default', 'Default'], loc = 'lower_right', bbox_to_anchor = (0.001, 0.001))
    plt.title(title + '\n', size = 15)
    plt.show()
    
def binary_classification_summary(model_name, dataset, model_instance, X, y, outcome_label = [0, 1]):
    """Prints summary of model outcome""" 
    dash_no = 220
    
    print ('\t' * 10 + '**** BINARY CLASSIFICATION RESULTS SUMMARY ****')    
    print ('-' * dash_no)
    
    ####
    print ('MODEL: {} | DATASET: {} | OUTCOME: {} vs {}'.format(model_name, dataset, outcome_label[0], outcome_label[1]))
    print ('TARGET DISTRIBUTION COUNT:')
    for i in range(len(outcome_label)):
        print (' ' * 3 + '{}: {} / {:0.3f}'.format(outcome_label[i], sum(y == np.unique(y)[i]), sum(y == np.unique(y)[i]) / len(y)))
    print ('-' * dash_no)     
    
    #####
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
    from sklearn.model_selection import cross_val_score    
    
    metrics_str = ['accuracy', 'recall', 'precision', 'f1']
    metrics_outcome = [np.round(cross_val_score(model_instance, X, y, scoring = x), 3) for x in metrics_str]
       
    print ('5 FOLD CROSS-VALIDATION METRIC SCORES WITH MEAN: \n')
    for metric, scores in zip(metrics_str, metrics_outcome):
        print ('{}: {} / {:0.3f}'.format(metric.upper(), scores, np.mean(scores), 3))
    
    scorer = make_scorer(recall_score, pos_label = 0)    
    specificity_array = np.round(cross_val_score(model_instance, X, y, scoring = scorer), 3)
    print ('SPECIFICITY: {} / {:0.3f}'.format(specificity_array, np.mean(specificity_array)))    
    print ('-' * dash_no)

    #####
    print ('CONFUSION MATRIX, ROC-AUC CURVE, PRECISION-RECALL CURVE [SINGLE RUN]: \n')
    fig = plt.figure(figsize = (30,5))
    #############
    ax = fig.add_subplot(141)
    
    from sklearn.metrics import confusion_matrix
    import itertools
    
    con_matrix = confusion_matrix(y, model_instance.predict(X))  
    cax = ax.matshow(con_matrix, cmap = plt.cm.OrRd)
    plt.grid(False)
    fig.colorbar(cax)

    perm = [p for p in itertools.product(range(2), repeat = 2)]
    cm_string = ['True Negatives', 'False Negatives', 'False Positives', 'True Positives']

    for label, i, value in zip(cm_string, perm, con_matrix.ravel()):
        plt.text(i[0], i[1], label + '\n' + str(value), va = 'center', ha = 'center', color = 'black' )

    plt.xlabel('Predicted'), plt.ylabel('Actual')
    plt.xticks([0,1], outcome_label), plt.yticks([0,1], outcome_label)

    #############
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
    
    ax = fig.add_subplot(142)
    fpr, tpr, thr = roc_curve(y, model_instance.predict_proba(X)[:, 1])
    auc = roc_auc_score(y, model_instance.predict_proba(X)[:, 1])

    plt.plot(fpr, tpr, label = 'AUC: {:0.3f}'.format(auc))
    plt.plot([0,1],[0,1], '--', color = 'black')
    
    count = 0
    no = round(len(y) / 60, -1)
    for i, j in zip(fpr, tpr):
        if count%no == 0:
            plt.annotate(round(thr[count], 3), xy = (i, j))
        count+=1     
    
    plt.xlim([-0.05, 1.05]), plt.ylim([-0.05, 1.05])
    plt.xlabel('(1 -  Specificity) a.k.a. FPR'), plt.ylabel('Sensitivity a.k.a TPR')
    plt.legend(loc = 'lower right'), plt.title('ROC-AUC curve')
    
    #############
    ax = fig.add_subplot(143)
    precision, recall, thresholds_pc = precision_recall_curve(y, model_instance.predict_proba(X)[:, 1])
    
    try:
        w = model_instance.decision_function(X)
    except AttributeError:
        w = [prob[1] for prob in model_instance.predict_proba(X)]
    AP = average_precision_score(y, w)
    plt.plot(recall, precision, label = 'Average Precision Score: {}'.format(round(AP, 3)))
    plt.legend(loc = 'lower right'), plt.title('Precision-Recall curve')
    plt.xlabel('Recall'), plt.ylabel('Precision')

    #######
    ax = fig.add_subplot(144)
    selected_threshold = float(input(' ' * 178 + 'SELECTED THRESHOLD:'))
    
    updated_predictions = [0 if x[1] <= selected_threshold else 1 for x in model_instance.predict_proba(X)]
    con_matrix_2 = confusion_matrix(y, updated_predictions)
    cax = ax.matshow(con_matrix_2, cmap = plt.cm.OrRd)
    plt.grid(False)
    fig.colorbar(cax)    
    
    for label, i, value in zip(cm_string, perm, con_matrix_2.ravel()):
        plt.text(i[0], i[1], label + '\n' + str(value), va = 'center', ha = 'center', color = 'black' )

    plt.xlabel('Predicted'), plt.ylabel('Actual')
    plt.xticks([0, 1], outcome_label), plt.yticks([0, 1], outcome_label)
    plt.show()
    
    print ('AUC GUIDELINES: [0.9 - 1.0 VERY GOOD | 0.8 - 0.9 G00D | 0.7 - 0.8 FAIR | 0.6 - 0.7 POOR | 0.5 - 0.6 FAIL]\nGENERAL THRESHOLD GUIDELINES: [LOWER % LEADS TO BETTER RECALL & POORER SPECIFICITY | VICE VERSA]')
    print ('-' * dash_no)

    #####
    print ('UPDATED METRICS USING SELECTED THRESHOLD OF {} [SINGLE RUN]:\n'.format(selected_threshold))

    original_scores = [np.mean(x) for x in metrics_outcome]
    updated_scores = [accuracy_score, recall_score, precision_score, f1_score]
    
    for metric, score_1, score_2 in zip(metrics_str, updated_scores, original_scores):
        print('{}: {} / DELTA: {:0.3f}'.format(metric.upper(), np.round(score_1(y, updated_predictions), 3), score_1(y, updated_predictions) - score_2))
        
    scorer = make_scorer(recall_score, pos_label = 0)
    updated_recall = recall_score(y, updated_predictions, pos_label = 0)
    print ('SPECIFICITY: {:0.3f} / DELTA: {:0.3f}'.format(updated_recall, updated_recall - np.mean(specificity_array)))
    print ('-' * dash_no)
    print ('FEATURE IMPORTANCES:')
    
    if X.shape[1] > 150:
        print ('Too many features to display')
    else:
        coef = pd.concat([pd.Series(X.columns).rename('Feature'), pd.Series(model_instance.coef_[0]).rename('coef')], axis = 1).sort_values('coef', ascending = False)
        fig, ax = plt.subplots(figsize = (30, 7))
        sns.barplot(x = coef.Feature, y = coef.coef)
        plt.title('Feature Importances')
        ax.xaxis.grid(True)
        plt.tick_params(axis = 'x', which = 'major', labelsize = 8)
        plt.xticks(rotation = 90)
        plt.show()

        from sklearn.preprocessing import MinMaxScaler

        coef['abs_coef'] = coef.coef.abs()

        coef['mm_coef'] = MinMaxScaler().fit_transform(coef.abs_coef.values[:, np.newaxis])
        coef.sort_values('mm_coef', ascending = False, inplace = True)

        fig, ax = plt.subplots(figsize = (30, 7))
        sns.barplot(x = coef.Feature, y = coef.mm_coef)
        plt.title('Scaled Feature Importances')
        ax.xaxis.grid(True)
        plt.yticks(np.linspace(0,1,41))
        plt.tick_params(axis='both', which='major', labelsize = 8)
        plt.xticks(rotation = 90)
        plt.show()

        mm_threshold = input('Select min-max coef. threshold:')
        to_remove = coef[coef.mm_coef <= float(mm_threshold)].Feature
        print('Number of suggested features to remove: {} \n\n{}'.format(len(to_remove), list(to_remove)))