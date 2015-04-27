"""
@author: efourrier

Purpose : This is a simple experimental class to detect outliers 


"""

import pandas as pd 
import numpy as np 

def cserie(serie):
	return serie[serie].index.tolist()

def iqr(ndarray):
	return np.percentile(ndarray,75) - np.percentile(ndarray,25)

def z_score(ndarray):
	return (ndarray - np.mean(ndarray))/(np.std(ndarray))

def iqr_score(ndarray):
	return (ndarray - np.median(ndarray))/(iqr(ndarray))

def mad_score(ndarray):
	return (ndarray - np.median(ndarray))/(np.median(np.absolute(ndarray -np.median(ndarray)))/0.6745)


class OutliersDetection(object):
	""" 
	this class focuses and identifying outliers

    Parameters
    ----------
    data : a pandas dataframe

    Examples
    --------
    * od = OutliersDetection(data = your_DataFrame)
    * cleaner.structure() : global structure of your DataFrame
    """

	def __init__(self,data):
		assert isinstance(data, pd.DataFrame)
		self.data = data
		self._nrow = len(self.data.index)
		self._ncol= len(self.data.columns)
		self._dfnumi = (self.data.dtypes == float)|(self.data.dtypes == int)
		self._dfnum = cserie(self._dfnumi)
		self._dfchari = (self.data.dtypes == object)
		self._dfchar = cserie(self._dfchari)

	@staticmethod
	def check_negative_value_serie(serie):
		""" this function will detect if there is negative value and calculate the 
		ratio negative value/postive value
		"""
		if serie.dtype == "object":
			TypeError("The serie should be numeric values")
		return sum(serie < 0)

	@staticmethod
	def outlier_detection_serie_d(serie,scores = [z_score,iqr_score,mad_score],
	cutoff_zscore = 3,cutoff_iqrscore = 2,cutoff_mad = 2):
		if serie.dtype == 'object':
		    raise("The variable is not a numeric variable")
		keys = [str(func.__name__) for func in scores]
		df = pd.DataFrame(dict((key,func(serie)) for key,func in zip(keys,scores)))
		df['is_outlier'] = 0
		if 'z_score' in keys :
		    df.loc[np.absolute(df['z_score']) >= cutoff_zscore,'is_outlier'] = 1
		if 'iqr_score' in keys :
		    df.loc[np.absolute(df['iqr_score']) >= cutoff_iqrscore,'is_outlier'] = 1
		if 'mad_score' in keys :
		    df.loc[np.absolute(df['mad_score']) >= cutoff_mad,'is_outlier'] = 1
		return df 

	def check_negative_value(self):
		""" this will return a the ratio negative/positve for each numeric 
		variable of the DataFrame
		"""
		return self.data[self._dfnum].apply(lambda x :self.check_negative_value_serie(x))

	def outlier_detection_d(self,subset = None,
	                    scores = [z_score,iqr_score,mad_score],
	                      cutoff_zscore = 3,cutoff_iqrscore = 2,cutoff_mad = 2):
	    """ Return a dictionnary with z_score,iqr_score,mad_score as keys and the 
	    associate dataframe of distance as value of the dictionnnary"""
	    df = self.data.copy()
	    numerc_variable = self._dfnum
	    if subset:
	        df = df.drop(subset,axis = 1)
	    df = df.loc[:,numerc_variable] # take only numeric variable 
	    # if remove_constant_col:
	    #     df = df.drop(self.constantcol(), axis = 1) # remove constant variable 
	    df_outlier = pd.DataFrame()
	    for col in df:
	        df_temp = self.outlier_detection_serie_d(df[col],scores,cutoff_zscore,
	            cutoff_iqrscore,cutoff_mad)
	        df_temp.columns = [col + '_' + col_name for col_name in df_temp.columns]
	        df_outlier = pd.concat([df_outlier,df_temp],axis = 1)
	    return df_outlier

