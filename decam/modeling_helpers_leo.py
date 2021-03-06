#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: efourrier

Purpose : This is a framework for Modeling with pandas, numpy and skicit-learn.
The Goal of this module is to rely on a dataframe structure for modelling g

"""


#########################################################
# Import modules and global helpers 
#########################################################

import pandas as pd 
import numpy as np 
from numpy.random import permutation
from sklearn.metrics import f1_score, precision_score, recall_score, explained_variance_score, r2_score

def cserie(serie):
	return serie[serie].index.tolist()


#########################################################
# Import modules for the Model class 
#########################################################
from sklearn import cross_validation


#########################################################
# DataCleaner class
#########################################################

class DataCleaner(object):
	""" 
	this class focuses abd identifying bad data as duplicated columns,rows,
	manymissing columns, long text data, keys of a dataframe, string aberation detection.

	For the most useful methods it will store the result into a attributes  

	It will also provide a way to fill na variables and provide type conversion 
	to transform the heterogenous structure of a pandas DataFrame into a 
	homogenous structure that is a numpy array.

	When you used a method the output will be stored in a instance attribute so you 
	don't have to compute the result again. 

    Parameters
    ----------
    data : a pandas dataframe
    label : a string naming the column of the output you want predict 

    Examples
    --------
    * cleaner = DataCleaner(data = your_DataFrame)
    * cleaner.structure() : global structure of your DataFrame
	* cleaner.psummary() to get the a global snapchat of the different stuff detected
	* data_cleaned = cleaner.basic_cleaning() to clean your data.
	"""


	def __init__(self, data):
		assert isinstance(data, pd.DataFrame)
		self.data = data
		# if not self.label:
		# 	print("""the label column is empty the data will be considered 
		# 		as a dataset of predictors""")
		self._nrow = len(self.data.index)
		self._ncol = len(self.data.columns)
		self._dfnumi = (self.data.dtypes == float) | (self.data.dtypes == int)
		self._dfnum = cserie(self._dfnumi)
		self._dfchari = (self.data.dtypes == object)
		self._dfchar = cserie(self._dfchari)
		self._nacolcount = pd.DataFrame()
		self._narowcount = pd.DataFrame()
		self._count_unique = pd.DataFrame()
		self._constantcol = []
		self._manymissingcol = []
		self._manymissingrow = []
		self._dupcol = []
		self._nearzerovar = pd.DataFrame()
		self._corrcolumns = []
		self._dict_info = {}
		self._structure = pd.DataFrame()
		self._string_info = ""

	# def get_label(self):
	# 	""" return the Serie of label you want predict """
	# 	if not self.label:
	# 		print("""the label column is empty the data will be considered 
	# 			as a dataset of predictors""")
	# 	return self.data[self.label]

	def count_unique(self):
	    """ Return a serie with the number of unique value per columns """
	    if len(self._count_unique):
	    	return self._count_unique
	    self._count_unique = self.data.apply(lambda x: x.nunique(), axis=0)
	    return self._count_unique


	def sample_df(self, pct=0.05, nr=10, threshold=None):
		""" sample a number of rows of a dataframe = min(max(0.05*nrow(self,nr),threshold)"""
		a = max(int(pct * float(len(self.data.index))), nr)
		if threshold:
			a = min(a, threshold)
		return self.data.loc[permutation(self.data.index)[:a]]

	def nacolcount(self):
		""" count the number of missing values per columns """
		if len(self._nacolcount):
			return self._nacolcount
		self._nacolcount = self.data.isnull().sum(axis=0)
		self._nacolcount = pd.DataFrame(self._nacolcount , columns=['Nanumber'])
		self._nacolcount['Napercentage'] = self._nacolcount['Nanumber'] / (self._nrow)
		return self._nacolcount

	def narowcount(self):
		""" count the number of missing values per columns """
		if len(self._narowcount):
			return self._narowcount
		self._narowcount = self.data.isnull().sum(axis=1)
		self._narowcount = pd.DataFrame(self._narowcount , columns=['Nanumber'])
		self._narowcount['Napercentage'] = self._narowcount['Nanumber'] / (self._nrow)
		return self._narowcount

	def manymissing(self, a=0.9, row=False):
		""" identify columns of a dataframe with many missing values ( >= a), if
		row = True row either.
		- the output is a list """
		if row:
			self._manymissingrow = self.narowcount()
			self._manymissingrow = cserie(self._manymissingrow['Napercentage'] >= a)
			return self._manymissingrow
		else :
			self._manymissingcol = self.nacolcount()
			self._manymissingcol = cserie(self._manymissingcol['Napercentage'] >= a)
			return self._manymissingcol

	def df_len_string(self):
		""" Return a Series with the max of the length of the string of string-type columns """
		return self.data.drop(self._dfnum, axis=1).apply(lambda x : np.max(x.str.len()), axis=0)

	def detectkey(self, index_format=False, pct=0.15, dropna=False, **kwargs):
	    """ identify id or key columns as an index if index_format = True or 
	    as a list if index_format = False """
	    if not dropna:
	        col_to_keep = self.sample_df(pct=pct, **kwargs).apply(lambda x: len(x.unique()) == len(x) , axis=0)
	        if len(col_to_keep) == 0:
	            return []
	        is_key_index = col_to_keep
	        is_key_index[is_key_index] == self.data.loc[:, is_key_index].apply(lambda x: len(x.unique()) == len(x) , axis=0)
	        if index_format:
	            return is_key_index
	        else :
	            return cserie(is_key_index)
	    else :
	        col_to_keep = self.sample_df(pct=pct, **kwargs).apply(lambda x: x.nunique() == len(x.dropna()) , axis=0)
	        if len(col_to_keep) == 0:
	            return []
	        is_key_index = col_to_keep
	        is_key_index[is_key_index] == self.data.loc[:, is_key_index].apply(lambda x: x.nunique() == len(x.dropna()), axis=0)
	        if index_format:
	            return is_key_index
	        else :
	            return cserie(is_key_index)

	def constantcol(self, **kwargs):
		""" identify constant columns """
		# sample to reduce computation time 
		if len(self._constantcol):
			return self._constantcol
		col_to_keep = self.sample_df(**kwargs).apply(lambda x: len(x.unique()) == 1, axis=0)
		if len(cserie(col_to_keep)) == 0:
			return []
		self._constantcol = list(cserie(self.data.loc[:, col_to_keep].apply(lambda x: len(x.unique()) == 1, axis=0)))
		return self._constantcol

	def factors(self, nb_max_levels=10, threshold_value=None, index=False):
	    """ return a list of the detected factor variable, detection is based on 
	    ther percentage of unicity perc_unique = 0.05 by default.
	    We follow here the definition of R factors variable considering that a 
	    factor variable is a character variable that take value in a list a levels

	    this is a bad implementation 


	    Arguments 
	    ----------
	    nb_max_levels: the mac nb of levels you fix for a categorical variable
	    threshold_value : the nb of of unique value in percentage of the dataframe length
	    index : if you want the result as an index or a list

	     """
	    if threshold_value:
	        max_levels = max(nb_max_levels, threshold_value * self._nrow)
	    else:
	        max_levels = nb_max_levels
	    def helper_factor(x, num_var=self._dfnum):
	        unique_value = set()
	        if x.name in num_var:
	            return False
	        else:
	            for e in x.values:
	                if len(unique_value) >= max_levels :
	                    return False
	                else:
	                    unique_value.add(e)
	            return True
	    

	    if index:
	        return self.data.apply(lambda x:  helper_factor(x))
	    else :
	        return cserie(self.data.apply(lambda x:  helper_factor(x)))


	def structure(self, threshold_factor=10):
		""" this function return a summary of the structure of the pandas DataFrame 
		data looking at the type of variables, the number of missing values, the 
		number of unique values """

		if len(self._structure):
			return self._structure
		dtypes = self.data.dtypes
		nacolcount = self.nacolcount()
		nb_missing = nacolcount.Nanumber
		perc_missing = nacolcount.Napercentage
		nb_unique_values = self.count_unique()
		dtypes_r = self.data.apply(lambda x: "character")
		dtypes_r[self._dfnumi] = "numeric"
		dtypes_r[(dtypes_r == 'character') & (nb_unique_values <= threshold_factor)] = 'factor'
		constant_columns = (nb_unique_values == 1)
		na_columns = (perc_missing == 1)
		is_key = nb_unique_values == self._nrow
		# is_key_na = ((nb_unique_values + nb_missing) == self.nrow()) & (~na_columns)
		dict_str = {'dtypes_r': dtypes_r, 'perc_missing': perc_missing,
		'nb_missing': nb_missing, 'is_key': is_key,
		'nb_unique_values': nb_unique_values, 'dtypes': dtypes,
		'constant_columns': constant_columns, 'na_columns': na_columns}
		self._structure = pd.concat(dict_str, axis=1)
		self._structure = self._structure.loc[:, ['dtypes', 'dtypes_r', 'nb_missing', 'perc_missing',
		'nb_unique_values', 'constant_columns', 'na_columns', 'is_key']]
		return self._structure


	def findupcol(self, threshold=100, **kwargs):
		""" find duplicated columns and return the result as a list of list """
		df_s = self.sample_df(threshold=100, **kwargs).T
		dup_index_s = (df_s.duplicated()) | (df_s.duplicated(take_last=True))
		
		if len(cserie(dup_index_s)) == 0:
			return []

		df_t = (self.data.loc[:, dup_index_s]).T
		dup_index = df_t.duplicated()
		dup_index_complet = cserie((dup_index) | (df_t.duplicated(take_last=True)))

		l = []
		for col in cserie(dup_index):
			index_temp = self.data[dup_index_complet].apply(lambda x: (x == self.data[col])).sum() == self._nrow
			temp = list(self.data[dup_index_complet].columns[index_temp])
			l.append(temp)
		self._dupcol = l 
		return self._dupcol


	def finduprow(self, subset=[]):
		""" find duplicated rows and return the result a sorted dataframe of all the
		duplicates
		subset is a list of columns to look for duplicates from this specific subset . 
		"""
		if sum(self.data.duplicated()) == 0:
			print("there is no duplicated rows")
		else: 
			if subset:
				dup_index = (self.data.duplicated(subset=subset)) | (self.data.duplicated(subset=subset, take_last=True)) 
			else :    
				dup_index = (self.data.duplicated()) | (self.data.duplicated(take_last=True))
				
			if subset :
				return self.data[dup_index].sort(subset)
			else :
				return self.data[dup_index].sort(self.data.columns[0])


	def nearzerovar(self, freq_cut=95 / 5, unique_cut=10, save_metrics=False):
	    """ identify predictors with near-zero variance. 
	            freq_cut: cutoff ratio of frequency of most common value to second 
	            most common value.
	            unique_cut: cutoff percentage of unique value over total number of 
	            samples.
	            save_metrics: if False, print dataframe and return NON near-zero var 
	            col indexes, if True, returns the whole dataframe.
	    """
	    nb_unique_values = self.count_unique()
	    percent_unique = 100 * nb_unique_values / self._nrow

	    def helper_freq(x):
	        if nb_unique_values[x.name] == 0:
	            return 0.0
	        elif nb_unique_values[x.name] == 1:
	            return 1.0
	        else:
	            return float(x.value_counts().iloc[0]) / x.value_counts().iloc[1] 

	    freq_ratio = self.data.apply(helper_freq)

	    zerovar = (nb_unique_values == 0) | (nb_unique_values == 1) 
	    nzv = ((freq_ratio >= freq_cut) & (percent_unique <= unique_cut)) | (zerovar)

	    if save_metrics:
	        return pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.data.columns)
	    else:
	        print(pd.DataFrame({'percent_unique': percent_unique, 'freq_ratio': freq_ratio, 'zero_var': zerovar, 'nzv': nzv}, index=self.data.columns))
	        return nzv[nzv == True].index 



	def findcorr(self, cutoff=.90, method='pearson', data_frame=False, print_mode=False):
		"""
		implementation of the Recursive Pairwise Elimination.        
		The function finds the highest correlated pair and removes the most 
		highly correlated feature of the pair, then repeats the process 
		until the threshold 'cutoff' is reached.
		
		will return a dataframe is 'data_frame' is set to True, and the list
		of predictors to remove oth		
		Adaptation of 'findCorrelation' function in the caret package in R. 
		"""
		res = []
		df = self.data.copy(0)
		cor = df.corr(method=method)
		for col in cor.columns:
			cor[col][col] = 0
		
		max_cor = cor.max()
		if print_mode:
			print(max_cor.max())
		while max_cor.max() > cutoff:            
			A = max_cor.idxmax()
			B = cor[A].idxmax()
			
			if cor[A].mean() > cor[B].mean():
				cor.drop(A, 1, inplace=True)
				cor.drop(A, 0, inplace=True)
				res += [A]
			else:
				cor.drop(B, 1, inplace=True)
				cor.drop(B, 0, inplace=True)
				res += [B]
			
			max_cor = cor.max()
			if print_mode:
				print(max_cor.max())
			
		if data_frame:
			return df.drop(res, 1)
		else:
			return res
			self._corrcolumns = res

	def psummary(self, manymissing_ph=0.70, manymissing_pl=0.05, nzv_freq_cut=95 / 5, nzv_unique_cut=10,
	threshold=100, string_threshold=40, dynamic=False):
		""" 
		This function will print you a summary of the dataset, based on function 
		designed is this package 
		- Output : python print 
		It will store the string output and the dictionnary of results in private variables 

		"""
		nacolcount_p = self.nacolcount().Napercentage
		if dynamic:
			print('there are {0} duplicated rows\n'.format(self.data.duplicated().sum()))
			print('the columns with more than {0:.2%} manymissing values:\n{1} \n'.format(manymissing_ph,
			cserie((nacolcount_p > manymissing_ph))))

			print('the columns with less than {0:.2%} manymissing values are :\n{1} \n you should fill them with median or most common value \n'.format(
			manymissing_pl, cserie((nacolcount_p > 0) & (nacolcount_p <= manymissing_pl))))

			print('the detected keys of the dataset are:\n{0} \n'.format(self.detectkey()))
			print('the duplicated columns of the dataset are:\n{0}\n'.format(self.findupcol(threshold=100)))
			print('the constant columns of the dataset are:\n{0}\n'.format(self.constantcol()))

			print('the columns with nearzerovariance are:\n{0}\n'.format(
			list(cserie(self.nearzerovar(nzv_freq_cut, nzv_unique_cut, save_metrics=True).nzv))))
			print('the columns highly correlated to others to remove are:\n{0}\n'.format(
			self.findcorr(data_frame=False)))
			print('these columns contains big strings :\n{0}\n'.format(
				cserie(self.df_len_string() > string_threshold)))
		else:
			self._dict_info = {'nb_duplicated_rows': sum(self.data.duplicated()),
						'many_missing_percentage': manymissing_ph,
						'manymissing_columns': cserie((nacolcount_p > manymissing_ph)),
						'low_missing_percentage': manymissing_pl,
						'lowmissing_columns': cserie((nacolcount_p > 0) & (nacolcount_p <= manymissing_pl)),
						'keys_detected': self.detectkey(),
						'dup_columns': self.findupcol(threshold=100),
						'constant_columns': self.constantcol(),
						'nearzerovar_columns' : cserie(self.nearzerovar(nzv_freq_cut, nzv_unique_cut, save_metrics=True).nzv),
						'high_correlated_col' : self.findcorr(data_frame=False),
						'big_strings_col': cserie(self.df_len_string() > string_threshold)
						} 

			self._string_info = u"""
there are {nb_duplicated_rows} duplicated rows\n
the columns with more than {many_missing_percentage:.2%} manymissing values:\n{manymissing_columns} \n
the columns with less than {low_missing_percentage:.2%}% manymissing values are :\n{lowmissing_columns} \n
you should fill them with median or most common value\n
the detected keys of the dataset are:\n{keys_detected} \n
the duplicated columns of the dataset are:\n{dup_columns}\n
the constant columns of the dataset are:\n{constant_columns}\n
the columns with nearzerovariance are:\n{nearzerovar_columns}\n
the columns highly correlated to others to remove are:\n{high_correlated_col}\n
these columns contains big strings :\n{big_strings_col}\n
			""".format(**self._dict_info)
			print(self._string_info)


	def basic_cleaning(self, manymissing_p=0.9, drop_col=None, filter_constantcol=True):
		""" 
	    Basic cleaning of the data by deleting manymissing columns, 
	    constantcol and drop_col specified by the user

	    """
		col_to_remove = []
		if manymissing_p:
			col_to_remove += list(self.manymissing(manymissing_p))
		if filter_constantcol:
			col_to_remove += list(self.constantcol())
		if isinstance(drop_col, list):
			col_to_remove += drop_col
		elif isinstance(drop_col, str):
			col_to_remove += [drop_col]
		else :
			pass
		return self.data.drop(pd.unique(col_to_remove), axis=1)

	@staticmethod
	def fillna_serie(serie, special_value=None):
		""" fill values in a serie default with the mean for numeric or the most common 
		factor for categorical variable """
		
		if special_value:
			return serie.fillna(serie.mean())
		if (serie.dtype == float) | (serie.dtype == int) :
			return serie.fillna(serie.mean())
		else:
			return serie.fillna(serie.value_counts().index[0])

	def fill_low_na(self, columns_to_process=[], threshold=None):
		""" this function will return a dataframe with na value replaced int 
		the columns selected by the mean or the most common value

		Arguments
		---------
		- columns_to_process : list of columns name with na values you wish to fill 
		with the fillna_serie function 

		Returns
		--------
		- a pandas Dataframe with the columns_to_process filled with the filledna_serie

		 """
		df = self.data
		if threshold:
			columns_to_process = columns_to_process + cserie(self.nacolcount().Napercentage < threshold)
		df.loc[:, columns_to_process] = df.loc[:, columns_to_process].apply(lambda x: self.fillna_serie(x))
		return df 

	def to_dummy(self, auto=False, auto_drop=False, include_na_dummy=True,
		subset=None, levels_limit=30, verbose=True):
		""" 
		this function will transform categorical variables to numeric dummy 
		variables so render a homogenous pandas DataFrame (only numeric variables)

		Arguments
		---------
		- auto : False if you want to disable the automatic transformation of the 
		factors variables, default False.

		-auto_drop : True if you want to automaticely drop character variables with 
		more levels that the levels_limit.

		- include_na_dummy : False if you don't want to include a default column for
		missing values.

		- subset : if you want add your own columns to transform, list of columns 
		name, default None.

		- levels_limit : drop the variable if the character variable has too many levels,
		default 30.

		- verbose : print if there is still non numeric variables in yout output,
		default True.

		Returns 
		--------
		a homogenous pandas DataFrame with only numeric variables
		"""
		df = self.data.copy()
		if auto:
			col_to_transform = self.factors(nb_max_levels=levels_limit)
			df = pd.get_dummies(df, columns=col_to_transform, dummy_na=True)
		if subset : 
			df = pd.get_dummies(df, columns=subset, dummy_na=True)
		if auto_drop: 
			df = df[(df.dtypes == float) | (df.dtypes == int)]
		if verbose : 
			if cserie(~((df.dtypes == float) | (df.dtypes == int))):
				print("There are still non numeric variables {0}".format(cserie(~((df.dtypes == float) | (df.dtypes == int)))))
		return df 

	def pandas_to_ndarray(self):
		"""
		Converts a dataframe to a homogenous ndarray and provides a function to help convert back
		to pandas object.

		Returns
		-------
		- a homogenous ndarray 
		- F : Function
		    F(Xvals) = X
		"""
		X = self.copy()
		return X.values, lambda arr: pd.DataFrame(arr, index=X.index, columns=X.columns)




#########################################################
# class Hybrid 
#########################################################

# class Hybrid(object):
# 	def __init__(self,data):
# 		self.data = data
# 		self._index = self.data.index
# 		self._column = self.data.columns

# 	def as_ndarray(self):
# 		return self.data.values

# 	def as_DataFrame(self,arr):
# 		return pd.DataFrame(self.data.index=self._index, columns=self._columns)





#########################################################
# Preprocessing and modeling class 
#########################################################


class Model(object):
	""" 
	This class is designed to help you doing predictive models and scoring with
	skicit-learn.

	Parameters
    ----------
    data : a homogenous numpy array without missing variables
    test : you can specify a test set 
    train : you can specify a train set 


    Examples
    --------
    """


	def __init__(self, my_array):
		assert isinstance(my_array, np.my_array)
		if np.isnan(my_array).any(): 
			raise("The array should not have missing value")
		self.my_array = my_array
		self._indices = None

	def build_indices(self, pct_split=None, nb_cv=None , nb_bootstrap=None ,
		bootstrap_pct=0.5, loocv=None, shuffle=True):
		""" this function will build a generator of index for the test and 
		training set 

		Arguments
		---------
		pct_split : the test percentage 
		nb_cv : the number of groups for cross validation 
		nb_bootstrap : True if you want activate bootstrap 
		loocv : True 

		Return
		-------
		a skicit-learn object which is a iterator with all the indexes 

		"""
		if pct_split:
			self._indices = cross_validation.ShuffleSplit(len(self.my_array),
			 n_iter=1, test_size=pct_split)
		if cv:
			self._indices = cross_validation.KFold(len(self.my_array), n_folds=nb_cv,
				shuffle=shuffle)
		if bootstrap:
			self._indices = cross_validation.Bootstrap(len(self.data), n_iter=nb_bootstrap,
				train_size=bootstrap_pct)

		if loocv:
			self._indices = cross_validation.LeaveOneOut(len(self.data))

		return self._indices
	
	def build_model(self, clf, clf_args=None, my_attr=list(), verbose=False, cv=False):
		""" this function will build a scikit-learn object and display information about the performance of the model if verbose is activated
		NB: build_indices is supposed to have been called before

		Arguments
		---------
		clf : sklearn classifier function
		clf_args : arguments for the sklearn classifier
		my_attr : attributes of the classifier specifically wanted by user
		verbose : boolean to enable display of results

		Return
		-------
		a skicit-learn classifier object

		"""
		
		if not cv and self._indices == None:
			raise("build_indices should be executed before build_model for cross-validation")
			return
		
		# create list of attributes of interest
		generic_attributes = ['feature_importances_', 'oob_score_', 'weights_']

		# add attributes wanted by user to the set of attributes
		set_of_attr = set(generic_attributes + my_attr)
		
		def my_cv_model(clf, clf_args, my_array, indices):
			
			# get data and predictions
			data, pred = my_array
			
			# get training and testing indices
			training_indices, testing_indices = indices
			
			# build training and testing data
			training_data = data[training_indices]
			training_pred = pred[training_indices]
			testing_data = data[testing_indices]
			testing_pred = pred[testing_indices]
		
			# initialize classifier with specified arguments
			if clf_args == None:
				my_clf = clf()
			else:
				my_clf = clf(**clf_args)

			# fit data
			my_clf.fit(training_data, training_pred)
			
			# get available attributes of classifier
			my_clf_dict = my_clf.__dict__
			
			# get attributes that are both of interest and available
			attr_dict = dict()
			for attr in my_clf_dict:
				if attr in set_of_attr:
					attr_dict[attr] = my_clf_dict[attr]
					
			# predict labels on training data
			res_dict = dict()
			my_pred = my_clf.predict(testing_data)
			try:
				for my_func in [f1_score, precision_score, recall_score]:
					res_dict[str(my_func)] = my_func(testing_pred, my_pred)
				self.model_type = "classification"
			except:
				res_dict['R2 score'] = r2_score(testing_pred, my_pred)
				self.model_type = "regression"
			
			# return result
			return res_dict
		
		def my_model(clf, clf_args, my_array):
			
			# get data and predictions
			data, pred = my_array
		
			# initialize classifier with specified arguments
			if clf_args == None:
				my_clf = clf()
			else:
				my_clf = clf(**clf_args)

			# fit data
			my_clf.fit(data, pred)
			
			# get available attributes of classifier
			my_clf_dict = my_clf.__dict__
			
			# get attributes that are both of interest and available
			attr_dict = dict()
			for attr in my_clf_dict:
				if attr in set_of_attr:
					attr_dict[attr] = my_clf_dict[attr]
			
			# return result
			return attr_dict
		
		# cross-validation case : estimating error
		if cv:
			
			# initialize dictionary of results
			self.res_dict = dict()
			for my_func in [f1_score, precision_score, recall_score]:
				self.res_dict[str(my_func)] = 0
			
			# get results for each set of indices of the cross-validation
			count = 0.	
			for training_indices, testing_indices in self._indices:
				my_indices = training_indices, testing_indices
				my_dict = my_cv_model(clf, clf_args, self.my_array, my_indices)
				for key in my_dict:
					self.res_dict[key] += my_dict[key]
				count += 1.
			
			# average results
			for key in self.res_dict:
				self.res_dict[key] /= count
			
			# print results if verbose
			if verbose:	
				print("-" * 50)
				print("Report for Regression")
				print("-" * 50)
				for res in self.res_dict:
					print(res + "\t" + str(self.res_dict[res]))
		
		# other case : building model on all data		
		else:
			
			# get model attributes
			self.attr_dict = my_model(clf, clf_args, self.my_array)
			
			# print results if verbose
			if verbose:
				print("-" * 50)
				print("Summary of attributes of interest")
				print("-" * 50)
				print("Attribute:" + "\t" + "Value")
				for attr in self.attr_dict:
					if attr in set_of_attr:
						print(attr + "\t" + str(self.attr_dict[attr]))
			
				
		return self._clf
		
		def get_train_test(self):
			if not self._buid :
				raise("build is empty you should choose your type of test/train splitting")
			return [(self.data[train], self.data[test]) for train, test in self._build]


