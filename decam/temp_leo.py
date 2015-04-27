from sklearn.metrics import classification_report, explained_variance_score, r2_score
from sklearn.linear_model import Lasso

import numpy as np

class Model:
    
    def __init__(self, data, labels, col_names):
        
        self.training_data = data
        self.training_labels = labels
        self.col_names = col_names
    
    def make_model(self, clf, args = None, verbose=False, add_attributes=list()):
        """
        initializes and trains model specified
        
        display variables of results and preliminary scores if verbose is set to True
        """
        
        # initialize classifier with specified arguments
        if args == None:
            self.clf = clf()
        else:
            self.clf = clf(**args)
        
        
        # fit data
        self.clf.fit(self.training_data, self.training_labels)
        
        # tell user
        if verbose:
            print "=" * 50
            print "Data fitted with model " + str(clf)
            print "=" * 50
        
        # create list of attributes of interest
        generic_attributes = ['feature_importances_', 'oob_score_', 'weights_']
        
        # add attributes wanted by user
        set_of_attr = set(generic_attributes + add_attributes)
        
        # get available attributes of classifier
        my_dict = self.clf.__dict__
        
        # print attributes that are both of interest and available    
        if verbose:
            print("-" * 50)
            print("Summary of attributes of interest")
            print("-" * 50)
            print("Attribute:" + "\t" + "Value")
            for attr in my_dict:
                if attr in set_of_attr:
                    print(attr + "\t" + str(my_dict[attr]))
        
        # predict labels on training data
        pred = self.clf.predict(self.training_data)

        # compute various scores of interest
        if verbose:
            try:
                # discrete output : classification
                my_report = classification_report(self.training_labels, pred, target_names=self.col_names)
                print("-" * 50)
                print("Report for Classification")
                print("-" * 50)
                print(my_report)
            
            except:
                # continuous output : regression
                print("-" * 50)
                print("Report for Regression")
                print("-" * 50)
                print("R2 Score: \t" + str(r2_score(self.training_labels, pred)))
                print("Explained Variance: \t" + str(explained_variance_score(self.training_labels, pred)))
                
                

data = list()
labels = list()    
n = 10    
for k in range(n):
    data += [[k]]
    labels += [[3 * k]]

data = np.array(data)
labels = np.array(labels)
    
my_model = Model(data, labels,["my_var"])
my_model.make_model(Lasso, verbose = True, add_attributes=['coef_', 'alpha'])