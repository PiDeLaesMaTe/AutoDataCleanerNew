# AutoDataCleanerNew (Teresa Piergili)
Cleans and preprocess the input variables in a Machine Learning model written in Python language. 


    
    """
    clean_me function performs automatic dataset cleaning to Pandas DataFrame as per the settings parameters passed to this function
    
    :param dataframe: input Pandas DataFrame on which the cleaning will be performed 
    :param detect_binray: if True, any column that has two unique values, these values will be replaced with 0 and 1 (e.g.: ['looser', 'winner'] => [0,1])
    :param numeric_dtype: if True, columns will be converted to numeric dtypes when possible **see [1] in README.md**
    :param decision_tree: if True, automatic detection of which variables will be transformed by decision tree and coded
    :param one_hot: if True, all non-numeric columns will be encoded to one-hot columns 
    :param normalize: if True, all non-binray (columns with values 0 or 1 are excluded) columns will be normalized and if True, automatic detection of which variables should be normalized 
    :param datetime_columns: a list of columns which contains date or time or datetime entries (important to be announced in this list, otherwise hot-encoding will mess up these columns)
    :param remove_columns: list of columns to remove, this is usually non-related featues such as the ID column 
    :param high_corr_elimination: eliminated high corr variables (Teresa)
    :param low_var_elimination: eliminates low variance variables (under 0.1) (Teresa)
    :param measuring_variable: The y of the model (Teresa)
    :param variable_to_encode: if decision_tree is true, variables that want to be changed by tree_decision
    :param outlier_removal: if outlier_removal is True, variables will be cleaned of the outliers
    :param duplicated_var: if duplicated_var is True, the variables will be eliminated 
    :param duplicated_rows_remove: if duplicated_rows_remove is True,the rows that are duplicated will be eliminated
    :param variables_uniques:if variables_uniques is True, the variables that are unique in all rows will be deleted
    :param clean: if clean is True, the data will be cleaned of NAN Values
    :param verbose: print progress in terminal/cmd
    :return: processed and clean Pandas DataFrame.
    """
    
    
