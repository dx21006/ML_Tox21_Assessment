def get_feature_importance(model_fitted, top_n=10, df_x_train=x_train): #need to provide a fitted model, top_n = top n features to return sorted by toxicity (default 10), df_x_train = the name of your x_train dataframe (for naming reasons). defaults to x_train
    """Returns the feature importances as a dataframe"""
    importances_permutation = model_fitted.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model_fitted.estimators_], axis=0)
    df_importances_permutation = pd.DataFrame(importances_permutation)
    df_importances_permutation['feature'] = df_x_train.columns

    df_top_n_importances_permutation = df_importances_permutation.nlargest(top_n, 0)
    print(df_top_n_importances_permutation,'\n')

    df_importances_permutation_FG = df_importances_permutation[df_importances_permutation['feature'].str.contains("fr_")] #new df containing only the features called fr_, i.e. the functional groups
    df_importances_permutation_FG_top_10 = df_importances_permutation_FG.nlargest(top_n,0)
    print(df_importances_permutation_FG_top_10)