import pandas as pd
from sklearn.feature_selection import chi2, f_classif
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_complement(sup:pd.DataFrame, sub:pd.DataFrame) -> pd.DataFrame:
    """
    Gets the rows in `sup` which aren't in `sub`.
    Assumes that `sub` is a subset of `sup` and the indices align.
    """
    sub_idx = list(sub.index)
    sup_idx = list(sup.index)
    complement_idx = []
    for i in sub_idx:
        sup_idx.remove(i)
        
    complement = sup.iloc[complement_idx, :]
    return complement


def stacked_hist(feature:str, df, target):
    plt.hist(x=[df.loc[train_df[target] == True,  feature],
                df.loc[train_df[target] == False, feature]],
             bins=200, stacked=True, color=['tab:orange','tab:blue'])
    plt.xlabel(feature)
    
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['tab:orange','tab:blue']]
    labels= [f'{target}: True', f'{target}: False']
    plt.legend(handles, labels)

    plt.show()


def get_chi2(X, y):
    """
    Returns a dictionary where the keys are the features and the values are tuples (chi2, p-value).
    """
    X_dummy = pd.get_dummies(X)
    stats = {}
    features = X.columns
    stats_list = chi2(X_dummy, y)

    for i in range(len(features)):
        stats[features[i]] = (stats_list[0][i], stats_list[1][i])

    return stats


def get_anova(X, y):
    """
    Returns a dictionary where the keys are the features and the values are tuples (chi2, p-value).
    """
    X_dummy = pd.get_dummies(X)
    stats = {}
    features = X.columns
    stats_list = f_classif(X_dummy, y)

    for i in range(len(features)):
        stats[features[i]] = (stats_list[0][i], stats_list[1][i])

    return stats


def univariate_summary(feature:str, df, target, chi2_stats:dict, anova_stats:dict, alpha:float=0.05):
    stacked_hist(feature, df=df, target='Attrition_Flag')

    try:
        print(f'Mean: {df[feature].mean()}')
        print(f'Standard Deviation: {df[feature].std()}')
        print(f'Minimum: {df[feature].min()}')
        print(f'Maximum: {df[feature].max()}\n')
    except:
        pass

    chi2_p = chi2_stats[feature][1]
    chi2_is_ind = chi2_p < alpha

    anova_p = anova_stats[feature][1]
    anova_is_ind = anova_p < alpha

    print(f'Chi^2 p-value: {chi2_p:.3f}')
    print(f'Chi^2: {feature} is probably independent of target: {chi2_is_ind}\n')

    print(f'ANOVA F-Test p-value: {anova_p:.3f}')
    print(f'ANOVA: {feature} is probably independent of target: {anova_is_ind}')
