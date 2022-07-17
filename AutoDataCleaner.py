import pandas as pd
import numpy as np 

#Charged libraries from Teresa 
from feature_engine import outliers
from feature_engine.selection import DropDuplicateFeatures
from feature_engine.selection import DropCorrelatedFeatures
from feature_engine.selection import DropConstantFeatures
from feature_engine.encoding import DecisionTreeEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression

def clean_me(dataframe, 
            detect_binary=True,
            numeric_dtype=True,
            decision_tree=True, 
            one_hot=True, 
            normalize=True,
            datetime_columns=[],
            remove_columns=[],
            high_corr_elimination=True,
            low_var_elimination=True,
            measuring_variable=[],
            variable_to_encode=[],
            outlier_removal=True,
            duplicated_var=True,
            duplicated_rows_remove=True,
            variables_uni=True,
            clean=True,
            verbose=True):
    
    
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
    
    df = dataframe.copy()


    if verbose:
        print(" +++++++++++++++ AUTO DATA CLEANING STARTED ++++++++++++++++ ")
        print('You have to choose if you want a tree encoder or just a numerical dtype (decision_tree=True)')
        print('You have to select the variable to eliminate such as name... (var_elimination=[' ',' '])')
        print('You have to select the variable to measure (Y) in (measuring_variable=[' ']) ')
        print('You have to choose which variables do you want to encode by decsion tree (variable_to_encode=[])')
        print('Check if the columns of date and time have been changed, if not, please introduce in datetime_columns=[''] the columns that need to be changed')


    # Validating user input to __init__ function 
    assert type(one_hot) == type(True), "Please ensure that one_hot param is bool (True/False)"
    # assert na_cleaner_mode in na_cleaner_modes, "Please choose proper value for naCleaner parameter. (Correct values are only: {})".format(self.__naCleanerOps)
    assert type(df) == type(pd.DataFrame()), "Parameter 'df' should be Pandas DataFrame"
    
    
    # Removing unwanted columns (usually its the ID columns...) ---------------------------------------------------------
    if len(remove_columns) > 0: 
        if verbose: 
            print(" =  AutoDataCleaner: Performing removal of unwanted columns... ")
        df = remove_columns_df(df,remove_columns,verbose)
        
        
    # Droping duplites (Teresa) --------------------------------------------------------------------------------------------
    if duplicated_rows_remove:
        if verbose:
            print(" = AutoDataCleaner: Performing removal of duplicated rows... ")
        df=duplicated_rows(df)
        
        
    # Droping variables uniques  (Teresa) --------------------------------------------------------------------------------------------    
    if variables_uniques:    
        if verbose:
            print(" = AutoDataCleaner: Performing recognition of unique variables and droping them...")
        df=variables_uniques(df,verbose)
    
    
    # Clean None (na) values (Teresa) --------------------------------------------------------------------------------------------
    if clean: 
        if verbose: 
            print(" =  AutoDataCleaner: Performing None/NA/Empty values cleaning... ")
        df=clean_nan(df,verbose)
        

    # Detecting binary columns ------------------------------------------------------------------------------------------
    if detect_binary:
        if verbose: 
            print(" =  AutoDataCleaner: Performing One-Hot encoding... ")
        df = detect_binary_df(df, verbose)
        
       
    # Decision tree encoder + Automatization (Teresa (Delete the parameter dtype))-------------------------------------------------------
    if decision_tree:  
        if len(variable_to_encode)>0: 
            if verbose:
                print(" =  AutoDataCleaner: Performing dataset decision tree encoder to the variable [{}] ".format(variable_to_encode)) 
            df=decision_tree_encoder(df,variable_to_encode,measuring_variable,verbose, datetime_columns)
        else: 
            if verbose: 
                print(" =  AutoDataCleaner: Performing dataset decision tree encoder to the variables  ...")          
            df=decision_tree_encoder(df,variable_to_encode,measuring_variable,verbose,datetime_columns)
               
                
    if numeric_dtype:
        if verbose: 
            print(" = AutoDataCleaner: Converting columns to numeric dtypes when possible ...")
        df=convert_numeric_df(df, datetime_columns, verbose)
        df=clean_nan(df,verbose)
        
    
    # Eliminate outliers (Teresa) -------------------------------------------------------------------------------------------------
    if outlier_removal:
        if verbose:
            print("=AutoDataCleaner: Performing Outliers removal... ")
    df=remove_outlier(df,verbose)
    

    # Casting datetime columns to datetime dtypes  -----------------------------------------------------------------------
    if len(datetime_columns)>0:
        if verbose: 
            print(" =  AutoDataCleaner: Casting datetime columns to datetime dtype... ")
        df = cols_to_datetime_dtype(df,datetime_columns,verbose)


    # Converting non-numeric to one hot encoding ------------------------------------------------------------------------
    if one_hot: 
        if verbose: 
            print(" =  AutoDataCleaner: Performing One-Hot encoding... ")
        cols_num_before =df.shape[1]
        df = one_hot_df(df)
        if verbose: 
            print("  + one-hot encoding done, added {} new columns".format(df.shape[1] - cols_num_before))


    # Eliminating columns with high corr (Teresa) -----------------------------------------------------------
    if high_corr_elimination: 
        if verbose: 
            print(" =  AutoDataCleaner: Performing dataset high correlation variables elimination...")
        df = drop_columns_HighCorr(df,verbose)
        
        
    # Eliminating Duplicated features  (Teresa) -----------------------------------------------------------
    if duplicated_var:
        if verbose:
            print(" =  AutoDataCleaner: Performing dataset droping duplicated features...")
        df=drop_duplicated_features(df,verbose)


    # Eliminating constant or nearly constant values (Teresa)-----------------------------------------------------------
    if low_var_elimination: 
        if verbose: 
            print(" =  AutoDataCleaner: Performing dataset droping constant or nearly constant  features...")
        df=drop_columns_constant_or_cuasi_constant(df,verbose)
        
        
    # Normalize all columns (binary 0,1 columns are excluded) AUTOMATIC (Teresa) -----------------------------------------------------------
    if normalize: 
        if verbose: 
            print(" =  AutoDataCleaner: Performing dataset normalization... ")
        df = normalize_df(df,measuring_variable,exclude=[datetime_columns,measuring_variable],verbose=True)

        
    if verbose: 
        print(" +++++++++++++++ AUTO DATA CLEANING FINISHED +++++++++++++++ ")


    return df



""" ------------------------------------------------------------------------------------------------------------------------- """

def datetime_dtype_series(series, verbose=True):
    """
    datetime_dtype_series function casts date columns to datetime dtype 

    :param df: input Pandas Series
    :returns: processed Pandas Series
    """
    try: 
        series = pd.to_datetime(series)
        if verbose: 
            print("  + converted column {} to datetime dtype".format(series.name))
        return series
    except Exception as e:
        print(" ERROR {}".format(e))


def cols_to_datetime_dtype(df, cols, verbose=True):
    """
    cols_to_datetime_dtype function casts given columns in dataframe to datetime dtype 

    :param df: input Pandas DataFrame
    :returns: processed Pandas DataFrame
    """
    for c in cols: 
        df[c] = datetime_dtype_series(df[c], verbose)
        if verbose: 
            print(" +  The following columns have been change [{}]".format(cols))
    return df


def remove_columns_df(df, remove_columns, verbose=True): 
    """
    remove_columns_df function removes columns in 'remove_columns' param list and returns df 
    
    :param df: input Pandas DataFrame 
    :param remove_columns: list of columns to be removed from the dataframe 
    :param verbose: print progress in terminal/cmd
    :returns: processed Pandas DataFrame 
    """
    stat = 0
    for col in remove_columns: 
        assert col in df.columns.to_list(), "{} is marked to be removed, but it does not exist in the dataset/dataframe".format(col)
        
        df.drop(columns=col, inplace=True)
        stat += 1
    if verbose: 
        print("  + removed {} columns successfully.".format(stat))
        print("  + removed the following columns: [{}] ".format(remove_columns))
    return df

def detect_binary_df(df, verbose=True): 
    """
    detect_binray function detects columns that has two unique values (e.g.: yes/no OR true/false etc...)
    and converts it to a boolean column containing 0 or 1 values only 
    :param df: input Pandas DataFrame 
    :param verbose: print progress in terminal/cmd
    :returns: processed Pandas DataFrame 
    """
    stat_cols = 0
    stat_cols_names = [] 
    stat_rows = 0
    for col in df.columns.to_list(): 
        # check if column has two unique values 
        if len(df[col].unique().tolist()) == 2: 
            unique_values = df[col].unique().tolist()
            unique_values.sort() # to ensure consistency during training and predicting
            df[col] = df[col].replace(unique_values[0], 0)
            df[col] = df[col].replace(unique_values[1], 1)
            stat_cols += 1
            stat_cols_names.append(col)
            stat_rows += df.shape[0]
    if verbose: 
        print("  + detected {} binary columns [{}], cells cleaned: {} cells".format(stat_cols, stat_cols_names, stat_rows))
    return df

def convert_numeric_series(series, force=False, verbose=True): 
    """
    convert_numeric_series function converts columns of dataframe to numeric dtypes when possible safely 
    if the values that cannot be converted to numeric dtype are minority in the series (< %25), then
    these minority values will be converted to NaN and the series will be forced to numeric dtype 
    :param series: input Pandas Series
    :param force: if True, values which cannot be casted to numeric dtype will be replaced with NaN 'see pandas.to_numeric() docs' (be careful with force=True)
    :param verbose: print progress in terminal/cmd
    :returns: Pandas series
    """
    stats = 0
    if force: 
        stats += series.shape[0]
        return pd.to_numeric(series, errors='coerce'), stats
    else: 
        # checking if values that cannot be converted to numeric are < 25% of entries in this series
        non_numeric_count = pd.to_numeric(series, errors='coerce').isna().sum()
        if non_numeric_count/series.shape[0] < 0.25: 
            # values that cannot be numeric are minority; hence, we set this as NaN and force that column to be
            # casted to numeric dtype, the 'clean_na_series' function will handle these NaN values laters 
            stats += series.shape[0]
            if verbose and non_numeric_count != 0: 
                print("  + {} minority (minority means < %25 of '{}' entries) values that cannot be converted to numeric dtype in column '{}' have been set to NaN, nan cleaner function will deal with them".format(non_numeric_count, series.name, series.name))
            return pd.to_numeric(series, errors='coerce'), stats
        else: 
            # this series probably cannot be converted to numeric dtype, we will just leave it as is
            return series, stats


def convert_numeric_df(df, exclude=[], force=False, verbose=True):
    """
    convert_numeric_df function converts dataframe columns to numeric dtypes when possible safely 
    if the values in a particular columns that cannot be converted to numeric dtype are minority in that column (< %25), then
    these minority values will be converted to NaN and the column will be forced to numeric dtype 
    :param df: input Pandas DataFrame
    :param exclude: list of columns to be excluded whice converting dataframe columns to numeric dtype (usually datetime columns)
    :param force: if True, values which cannot be casted to numeric dtype will be replaced with NaN 'see pandas.to_numeric() docs' (be careful with force=True)
    :param verbose: print progress in terminal/cmd
    :returns: Pandas DataFrame
    """
    stats = 0
    for col in df.columns.to_list(): 
        if col in exclude:
            continue
        df[col], stats_temp = convert_numeric_series(df[col], force, verbose)
    stats += stats_temp
    if verbose: 
        print("  + converted {} cells to numeric dtypes".format(stats))
    return df  


def one_hot_df(df): 
    """ 
    one_hot_df returns one-hot encoding of non-numeric columns in all the columns of the passed Pandas DataFrame
    
    :param df: input Pandas DataFrame
    :returns: Pandas DataFrame
    """
    return pd.get_dummies(df)

"""
def clean_na_series(series, na_cleaner_mode): 
    
    clean_nones function manipulates None/NA values in a given panda series according to cleaner_mode parameter
        
    :param series: the Panda Series in which the cleaning will be performed 
    :param na_cleaner_mode: what cleaning technique to apply, 'na_cleaner_modes' for a list of all possibilities 
    :returns: cleaned version of the passed Series

    if na_cleaner_mode == 'remove row': 
        return series.dropna()
    elif na_cleaner_mode == 'mean':
        mean = series.mean()
        return series.fillna(mean)
    elif na_cleaner_mode == 'mode':
        mode = series.mode()[0]
        return series.fillna(mode)
    elif na_cleaner_mode == False: 
        return series
    else: 
        return series.fillna(na_cleaner_mode)


def clean_na_df(df, na_cleaner_mode, verbose=True): 

    clean_na_df function cleans all columns in DataFrame as per given na_cleaner_mode
    
    :param df: input DataFrame
    :param na_cleaner_mode: what technique to apply to clean na values 
    :param verbose: print progress in terminal/cmd
    :returns: cleaned Pandas DataFrame 
    
    stats = {}
    for col in df.columns.to_list(): 
        if df[col].isna().sum() > 0: 
            stats[col + " NaN Values"] = df[col].isna().sum()
            try:
                df[col] = clean_na_series(df[col], na_cleaner_mode)
            except: 
                pass
            print("  + could not find mean for column {}, will use mode instead to fill NaN values".format(col))
            df[col] = clean_na_series(df[col], 'mode')
    if verbose: 
        print("  + cleaned the following NaN values: {}".format(stats))
    return df
"""
def normalize_df(data,mea_var, exclude=[], verbose=True): 
    """
    normalize_df function performs normalization to all columns of dataframe excluding binary (1/0) columns 
    
    :param df: input Pandas DataFrame
    :param exclude: list of columns to be excluded when performing normalization (usually datetime columns)
    :param verbose: print progress in terminal/cmd
    :returns: normalized Pandas DataFrame 
    """
    stats = 0
    var=[]
    for col in data.columns.to_list(): 
        if col in exclude: 
            continue
        # check if column is binray
        col_unique = data[col].unique().tolist()
        if len(col_unique) == 2 and 0 in col_unique and 1 in col_unique: 
            continue
        else:
            columna=deciding_normalization_variables(data,col,mea_var)
            if columna:
                var.append(col)
            else:
                print('The variable [{}] will not be normalized because its accuracy is too low'.format(col))  
        for col in data.columns.to_list():
            if col in var: 
                data[col] = (data[col]-data[col].mean())/data[col].std()
                stats += data.shape[0]
                columns=[data[col].name]
                if verbose:
                    print("  + normalized [{}]".format(columns))
                if verbose: 
                    print("  + normalized {} cells".format(stats))
    return data



def help(): 
    help_text = """
    ++++++++++++++++++ AUTO DATA CLEANER HELP +++++++++++++++++++++++
    FUNCTION CALL:
    AutoDataCleaner.clean_me(df, one_hot=True, na_cleaner_mode="mean", normalize=True, remove_columns=[], verbose=True)

    FUNCTION PARAMETERS:
    df: input Pandas DataFrame on which the cleaning will be performed 
    one_hot: if True, all non-numeric columns will be encoded to one-hot columns 
    na_cleaner_mode: what technique to use when dealing with None/NA/Empty values. Modes: 
        False: do not consider cleaning na values 
        'remove row': removes rows with a cell that has NA value
        'mean': substitues empty NA cells with the mean of that column  
        'mode': substitues empty NA cells with the mode of that column
        '*': any other value will substitute empty NA cells with that particular value passed here 
    normalize: if True, all non-binray (columns with values 0 or 1 are excluded) columns will be normalized. 
    remove_columns: list of columns to remove, this is usually non-related featues such as the ID column 
    verbose: print progress in terminal/cmd
    returns: processed and clean Pandas DataFrame 
    ++++++++++++++++++ AUTO DATA CLEANER HELP +++++++++++++++++++++++
    """
    print(help_text)


# Creation of new funtions (Teresa)  -----------------------------------------------------------------------------------------------------


def remove_outlier(df,verbose):
    capper = outliers.OutlierTrimmer(capping_method = "gaussian" , 
                                 tail='both', fold=3 ,missing_values="ignore")
    df=capper.fit_transform(df)
    maximum=capper.right_tail_caps_
    minimum=capper.left_tail_caps_
    if verbose: 
        print("+ The following [{}] shows dictionary with the maximum values above which values will be removed ".format(maximum))
        print("+ The following [{}] shows dictionary with the minimum values below which values will be removed ".format(minimum))
    return df


"""
#Elimination of nan values 

Copyright (c) 2016 Randal S. Olson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


# Clean nan (removing the column if 60% is nan, removing 5% if the line is nan and mode or mean depending on type of variables )


def clean_nan(df,verbose):
    #1. Drop the col if there are 60% of values missing 
    if verbose:
        print('Inside the funtion for cleaning variable, we proceed to eliminate the data with more than 60% of values beeing missing values')
    perc = 60.0
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    null_percentage = df.isnull().sum()/df.shape[0]*100
    col_to_drop=null_percentage[null_percentage>perc].keys()
    df = df.dropna( axis=1,thresh=min_count)
    if verbose: 
        print(" + the following columns '{}' had {}% or more of the values being missing ".format(col_to_drop,perc))
    #2. Drop the line if there is more than 5% of values missing (95% is preserved)
    if verbose:
        print('Inside the funtion for cleaning variable, we proceed to eliminate the row with more than 5% of values beeing missing values')
    perc=5.0
    df =df[df.isnull().sum(axis=1) < perc/100*len(df)]
    null_percentage = df.isnull().sum(axis=1)/df.shape[1]*100
    line_to_drop= null_percentage[null_percentage>perc].keys()
    if verbose: 
        print(" + the following rows '{}' had 5% or more of the values being missing ".format(line_to_drop))
    #3. Replace NaNs with the median or mode of the column depending on the column type
    if verbose:
        print('Inside the funtion for cleaning variable, we proceed to replace the missing values for the median/mode depending on type of varibles')
    for column in df.columns.values:
    # Replace NaNs with the median or mode of the column depending on the column type
        try:
            df[column].fillna(df[column].median(), inplace=True)
            median_col=df[column].name
            if verbose:
                print(" + The following variables '{}' have been fill by the median".format(median_col))
        except TypeError:
            most_frequent = df[column].mode()
        # If the mode can't be computed, use the nearest valid value
        # See https://github.com/rhiever/datacleaner/issues/8
            if len(most_frequent) > 0:
                df[column].fillna(df[column].mode()[0], inplace=True)
                mode_col=df[column].name
                if verbose:
                    print(" + The following variables '{}' have been fill by the mode".format(mode_col))
                    
            else:
                df[column].fillna(method='bfill', inplace=True)
                df[column].fillna(method='ffill', inplace=True)
    return df



# Elimination of columns with high correlation over 0.8


def drop_columns_HighCorr(df,verbose):
    tr = DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.8)
    df=tr.fit_transform(df)
    #Before running this code, you should run the code for DropCorrelatedFeatures for it to understand self.correlated_features_sets
    correlete=tr.correlated_feature_sets_
    if verbose: 
        print("  + droping [{}] cells".format(correlete))
    return df


# Elimination of columns with little variance (constant or cuasi constant 90%)


def drop_columns_constant_or_cuasi_constant(df,verbose):
    trans= DropConstantFeatures(tol=0.9)
    df=trans.fit_transform(df)
    const=trans.features_to_drop_
    if verbose: 
        print("  +  droping [{}] cells".format(const))
    return df


# Elimination of duplicated features 


def drop_duplicated_features(df,verbose):

    transformer = DropDuplicateFeatures()
    # fit and transform the data 
    df=transformer.fit_transform(df)
    dup=transformer.duplicated_feature_sets_
    if verbose: 
        print("  + droping [{}] cells".format(dup))
    return df


# Desiton tree encoder 


def decision_tree_encoder(data,var,y,verbose,datetime_columns):
    # set up the encoder
    name=[]
    if len(var)>0:
        print("  + The following variables are beeing transforme by decision tree encoder [{}]".format(var))
        encoder = DecisionTreeEncoder(variables=var,regression=False, random_state=0)
        data=encoder.fit_transform(data,data[y])
    else: 
        var=[]
        for col in data.columns:
            if (data[col].dtype==np.object):
                name.append(col)
                            
    for col in name:
        columna=deciding_tree_variables(data,col,datetime_columns,y,verbose)
        if columna:
            if col in columna: 
                var.append(col)
        else:
            print('The variable [{}] will not be converted to decision tree encoder because its accuracy is too low'.format(col))
    if len(var)==len(name):
        encoder = DecisionTreeEncoder(variables=var,regression=False, random_state=0)
        data=encoder.fit_transform(data,data[y])
        print("  + the variables that are beeing transformed by the automatic decision tree encoder are [{}] ".format(var))
    else:
        data=convert_numeric_df(data,var, verbose=False)
        data= clean_nan(data,verbose=False)
        encoder = DecisionTreeEncoder(variables=var,regression=False, random_state=0)
        data=encoder.fit_transform(data,data[y])
        print("  + the variables that are beeing transformed are [{}] ".format(var))
    return data


# dt_auto


def dt_inplace(df):
    """Automatically detect and convert (in place!) each
    dataframe column of datatype 'object' to a datetime just
    when ALL of its non-NaN values can be successfully parsed
    by pd.to_datetime().  Also returns a ref. to df for
    convenient use in an expression.
    """
    from pandas.errors import ParserError
    for c in df.columns[df.dtypes=='object']: #don't cnvt num
        try:
            df[c]=pd.to_datetime(df[c])
        except (ParserError,ValueError): #Can't cnvrt some
            pass # ...so leave whole column as-is unconverted
    return df
def read_csv(*args, **kwargs):
    """Drop-in replacement for Pandas pd.read_csv. It invokes
    pd.read_csv() (passing its arguments) and then auto-
    matically detects and converts each column whose datatype
    is 'object' to a datetime just when ALL of the column's
    non-NaN values can be successfully parsed by
    pd.to_datetime(), and returns the resulting dataframe.
    """
    return dt_inplace(pd.read_csv(*args, **kwargs))


def read_csv_transformado():
    text = """
    ++++++++++++++++++ AUTO DATA CLEANER READ CSV TRANSFORMADO +++++++++++++++++++++++
    OPTION 1: 
        IMPORT LIBRARY: 
            import pandas as pd 
        FUNCTION CALL:
            df=pd.read_csv('mydata.csv')
            
        DO NOT FORGET TO INCLUDE IN datetime_columns=[] , the variables date/time that
        need to be changed. 
    OPTION 2: 
    IMPORT LIBRARY:
        from dt_auto import read_csv 
    FUNCTION CALL:
        df=read_csv('mydata.csv')

    FUNCTION 
    import pandas as pd
    def dt_inplace(df):
        #Automatically detect and convert (in place!) each
        #dataframe column of datatype 'object' to a datetime just
        #when ALL of its non-NaN values can be successfully parsed
        #by pd.to_datetime().  Also returns a ref. to df for
        #convenient use in an expression.
        
        from pandas.errors import ParserError
        for c in df.columns[df.dtypes=='object']: #don't cnvt num
            try:
                df[c]=pd.to_datetime(df[c])
            except (ParserError,ValueError): #Can't cnvrt some
                pass # ...so leave whole column as-is unconverted
        return df
    def read_csv(*args, **kwargs):
        #Drop-in replacement for Pandas pd.read_csv. It invokes
        #pd.read_csv() (passing its arguments) and then auto-
        #matically detects and converts each column whose datatype
        #is 'object' to a datetime just when ALL of the column's
        #non-NaN values can be successfully parsed by
        #pd.to_datetime(), and returns the resulting dataframe.

        return dt_inplace(pd.read_csv(*args, **kwargs))
    DO NOT FORGET TO REMOVE THE # FROM EVERY LINE
    ++++++++++++++++++ AUTO DATA CLEANER HELP +++++++++++++++++++++++
    """
    print(text)



def deciding_normalization_variables(X,cl,y):
    X[cl] = (X[cl]-X[cl].mean())/X[cl].std()
    X_train, X_test, y_train, y_test = train_test_split(X,X[y], random_state=42, train_size = .8)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    preds=clf.predict(X_test)
    score_preds=(metrics.accuracy_score(y_test,preds))*100
    if score_preds>=80:
        colum=cl
        print('The column [{}] is transformed which produces an accuracy in the model of {} which is over 80%'.format(colum,score_preds))
        return colum
    
def variables_uniques(X,verbose):
    for col in X.columns:
        count=len(X[col].unique())
        if count==len(X[col].index):
            X.drop(columns=col, inplace=True)
            if verbose: 
                print('Droping [{}] columsn because they contained {} unique values '. format(col,count))
        else:
            print('Feature [{}] is not unique '.format(col))
            data=X
    return data



def deciding_tree_variables(X,cl,datetime_columns,y,verbose):
    encoder = DecisionTreeEncoder(variables=cl,regression=False, random_state=42)
    encoder.fit(X,X[y])
    X=encoder.transform(X)
    X=convert_numeric_df(X, datetime_columns,verbose)
    X = clean_nan(X,verbose=False)
    X_train, X_test, y_train, y_test = train_test_split(X,X[y], random_state=0, train_size = .8)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    preds=clf.predict(X_test)
    score_preds=(metrics.accuracy_score(y_test,preds))*100
    
    if score_preds>=80:
        colum=cl
        print('The column [{}] is transformed by decision tree encoder encoder which produces an accuracy in the model of {} which is over 80%'.format(colum,score_preds))
        return colum

def duplicated_rows(df):
    data=df.drop_duplicates()
    count=len(df)-len(data)
    print('Duplicated columns {}'.format(count))
    print(data)
    return data
