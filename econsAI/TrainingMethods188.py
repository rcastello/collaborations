from __future__ import division

from numpy import *
from pylab import *
from scipy import *
from pandas import *
import random

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('ggplot')
sns.set(style="ticks", font_scale=1.2, context="talk")
sns.set_style("white", {'axes.grid' : True})
plt.rcParams['figure.figsize'] = (15, 7)

#from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor,RandomForestClassifier
from sklearn.metrics import confusion_matrix
from time import time
import os


class CustomSVC:

    def __init__(self,X,Y = None):
        '''
        Y is a self variable for the classical training, but not for month or array training functions,
        so we put it no Y as the default value if it is not expressed.
        '''
        self.Xlabel = X
        self.Ylabel = Y

    def TrainAndTest(self, kernel):

        Boundtrain= round(0.75*len(self.Xlabel))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        #################################

        if kernel== "rbf":
            C_range = 2. ** np.arange(-5, 15)
            gamma_range = 2. ** np.arange(-10, 5)

            param_grid = dict(gamma=gamma_range, C=C_range)
            # CV with gridsearch and CV
            clf = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, n_jobs=-1,cv=6)

        if kernel== "linear":
            C_range = 2. ** np.arange(-5, 15)
            param_grid = dict(C=C_range)
            # CV with gridsearch and CV
            clf = GridSearchCV(SVC(kernel=kernel), param_grid=param_grid, n_jobs=-1,cv=6)


        clf.fit(Xtrn, Ytrn)
        print("The best parameters are : ", clf.best_params_)


        if kernel== "rbf":
            svc_al = SVC(kernel=kernel, C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
        if kernel== "linear":
            svc_al = SVC(kernel=kernel, C=clf.best_params_['C'])

        svc_al.fit(Xtrn, Ytrn)

        Ypredicted = svc_al.predict(Xtest)

        # # try it on the validation test
        # rmse_svr = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        # RRMSE_svr = rmse_svr / ( sum(Ytest)/len(Ytest))
        # #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        # print ('Test rmse: %.4f' % rmse_svr)
        # print ('Test Rrmse: %.4f' % RRMSE_svr)
        accuracy = float(sum(Ypredicted==Ytest))/float(len(Ytest))

        confMat = confusion_matrix(Ytest,Ypredicted)
        AccPerClass = diag(confMat)/(confMat.sum(axis=1))
        AccPerClassRound = array([round(el,2) for el in  AccPerClass*100])
        confMatWithAcc = vstack([ confMat.T,AccPerClassRound.T ]).T

        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test accuracy for RF: %.4f' % accuracy)
        # the values in the last column of the matrix are individual class accuracies, in percentage (10 for 10%)
        print(confMatWithAcc)

        return(svc_al,accuracy,confMatWithAcc)

class CustomSVR:

    def __init__(self,X,Y = None):
        '''
        Y is a self variable for the classical training, but not for month or array training functions,
        so we put it no Y as the default value if it is not expressed.
        '''
        self.Xlabel = X
        self.Ylabel = Y

    def TrainAndTest(self):

        Boundtrain= round(0.75*len(self.Xlabel))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        #################################
        C_range = 2. ** np.arange(-5, 15)
        gamma_range = 2. ** np.arange(-10, 5)
        eps_range = 2. ** np.arange(-5, 5)
        param_grid = dict(gamma=gamma_range, C=C_range,epsilon=eps_range)
        #clf = GridSearchCV(SVR(), param_grid=param_grid, cv=StratifiedKFold(y=Xtrn, n_folds=6))
        clf = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1,cv=6)
        clf.fit(Xtrn, Ytrn)
        print("The best parameters are : ", clf.best_params_)
        svr_al=SVR(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'], epsilon=clf.best_params_['epsilon'])
        svr_al.fit(Xtrn, Ytrn)
        Ypredicted=svr_al.predict(Xtest)

        # try it on the validation test
        rmse_svr = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_svr = rmse_svr / ( sum(Ytest)/len(Ytest))
        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test rmse: %.4f' % rmse_svr)
        print ('Test Rrmse: %.4f' % RRMSE_svr)

        return(svr_al,rmse_svr,RRMSE_svr,clf.best_params_['C'], clf.best_params_['gamma'], clf.best_params_['epsilon'])


    def Fast_TrainAndTest(self,C,gamma,eps):
        Boundtrain= round(0.75*len(self.Xlabel))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]
        #################################
        svr_al=SVR(kernel='rbf', C=C, gamma=gamma, epsilon=eps)
        svr_al.fit(Xtrn, Ytrn)
        Ypredicted=svr_al.predict(Xtest)
        # try it on the test test
        rmse_svr = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_svr = rmse_svr / ( sum(Ytest)/len(Ytest))
        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test rmse: %.4f' % rmse_svr)
        print ('Test Rrmse: %.4f' % RRMSE_svr)
        #('The best parameters are : ', {'epsilon': 0.03125, 'C': 16384.0, 'gamma': 0.25})
        #Test rmse for SVR2D: 34995.2713
        return(svr_al,rmse_svr,RRMSE_svr,C, gamma, eps)


    def Monthly_TrainAndTest(self,Ymonths):
        '''
        Train and test in the case of monthly data.
        Ymonths is the matrix of monthly output values, (N by 12 matrix).
        '''
        Boundtrain= round(0.75*len(self.Xlabel))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],Ymonths[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],Ymonths[Boundtrain:]
        SVRlist,rmselist,RRMSElist = [],[],[]
        mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        #################################
        CList,gammaList,epsList=[],[],[]
        for i in range(len(mon)):

            C_range = 2. ** np.arange(-5, 15)
            gamma_range = 2. ** np.arange(-10, 5)
            eps_range = 2. ** np.arange(-5, 5)
            param_grid = dict(gamma=gamma_range, C=C_range,epsilon=eps_range)
            #clf = GridSearchCV(SVR(), param_grid=param_grid, cv=StratifiedKFold(y=Ztrn, n_folds=5))
            clf = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1,cv=6)
            clf.fit(Xtrn, Ytrn[:,i])
            print("The best parameters are : ", clf.best_params_)

            SVRlist.append( SVR(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'],epsilon=clf.best_params_['epsilon']) )
            SVRlist[i].fit(Xtrn, Ytrn[:,i])
            Zpredicted=SVRlist[i].predict(Xtest)
            # try it on the validation test
            rmselist.append( sqrt(sum((Zpredicted-Ytest[:,i])**2)/len(Ytest[:,i]))  )
            RRMSElist.append( rmselist[i] / ( sum(Ytest[:,i])/len(Ytest[:,i])) )
            #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
            print(mon[i])
            print ('Test rmse: %.4f' % rmselist[i])
            print ('Test Rrmse: %.4f' % RRMSElist[i])
            CList.append(clf.best_params_['C'])
            gammaList.append(clf.best_params_['gamma'])
            epsList.append(clf.best_params_['epsilon'])

        #('The best parameters are : ', {'epsilon': 0.03125, 'C': 16384.0, 'gamma': 0.25})
        #Test rmse for SVR2D: 34995.2713
        return(SVRlist,rmselist,RRMSElist,CList,gammaList,epsList)


    def Array_TrainAndTest(self,Yarray):
        '''
        Generalization of SVR_month with any kind of Yarray,
        for example to predict frequencies for slope and azimuth

        '''
        #Yarray is the matrix of monthly output values, (N by 12 matrix)
        Boundtrain = round(0.75*len(self.Xlabel))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],Yarray[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],Yarray[Boundtrain:]
        SVRlist,rmselist, RRMSElist = [],[],[]
        steps = [i for i in range(Yarray.shape[1])]

        #################################
        CList,gammaList,epsList=[],[],[]
        for i in range(Yarray.shape[1]):

            C_range = 2. ** np.arange(-5, 15)
            gamma_range = 2. ** np.arange(-10, 5)
            eps_range = 2. ** np.arange(-5, 5)
            param_grid = dict(gamma=gamma_range, C=C_range,epsilon=eps_range)
            #clf = GridSearchCV(SVR(), param_grid=param_grid, cv=StratifiedKFold(y=Ztrn, n_folds=5))
            clf = GridSearchCV(SVR(), param_grid=param_grid, n_jobs=-1,cv=6)
            clf.fit(Xtrn, Ytrn[:,i])
            print("The best parameters are : ", clf.best_params_)

            SVRlist.append( SVR(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'],epsilon=clf.best_params_['epsilon']) )
            SVRlist[i].fit(Xtrn, Ytrn[:,i])
            Zpredicted=SVRlist[i].predict(Xtest)
            # try it on the validation test
            rmselist.append( sqrt(sum((Zpredicted-Ytest[:,i])**2)/len(Ytest[:,i]))  )
            RRMSElist.append( rmselist[i] / ( sum(Ytest[:,i])/len(Ytest[:,i])) )
            #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
            print(steps[i])
            print ('Test rmse: %.4f' % rmselist[i])
            print ('Test Rrmse: %.4f' % RRMSElist[i])

            CList.append(clf.best_params_['C'])
            gammaList.append(clf.best_params_['gamma'])
            epsList.append(clf.best_params_['epsilon'])

        #('The best parameters are : ', {'epsilon': 0.03125, 'C': 16384.0, 'gamma': 0.25})
        #Test rmse for SVR2D: 34995.2713
        return(SVRlist,rmselist,RRMSElist, CList,gammaList,epsList)


class CustomRF:

    def __init__(self,X,Y = None):
        self.Xlabel = X
        self.Ylabel = Y



    def Fast_TrainAndTest(self, n_trees, m,min_samples_leaf=3):
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            # test set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        rf = RandomForestRegressor(n_estimators=n_trees,
                                     oob_score=True,
                                     max_depth= None,
                                     max_features= m ,
                                     min_samples_split=2,
                                    min_samples_leaf = min_samples_leaf)
        rf.fit(Xtrn, Ytrn)
        Ypredicted = rf.predict(Xtest)

        Ypredicted_training = rf.predict(Xtrn)

        # try it on the validation test
        rmse_rf = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_rf = rmse_rf / ( sum(Ytest)/len(Ytest))

        # try it on the TRAINING test
        rmse_rf_train = sqrt(   sum((Ypredicted_training-Ytrn)**2) /len(Ytrn) )
        RRMSE_rf_train = rmse_rf_train / ( sum(Ytrn)/len(Ytrn))

        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test rmse: %.4f' % rmse_rf)
        print ('Test Rrmse: %.4f' % RRMSE_rf)
        print ('Training Rrmse: %.4f' % RRMSE_rf_train)
        print ('OOB score: %.4f' % rf.oob_score_)
        print('salut')
        #Test rmse for SVR2D: 34995.2713
        return(rf,rmse_rf,RRMSE_rf,rf.oob_score_)

    def FastMonthly_TrainAndTest(self, n_trees, m, Ymonths):
        '''
        Train and test in the case of monthly data.
        Ymonths is the matrix of monthly output values, (N by 12 matrix).
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],Ymonths[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],Ymonths[Boundtrain:]

        RFlist,rmselist,RRMSElist,oob_scorelist = [],[],[],[]
        mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

        #################################
        for i in range(len(mon)):

            RFlist.append( RandomForestRegressor(n_estimators=n_trees,
                                     oob_score=True,
                                     max_depth= None,
                                     max_features= m ,
                                     min_samples_split=2,
                                    min_samples_leaf = 3) )
            RFlist[i].fit(Xtrn, Ytrn[:,i])
            Zpredicted=RFlist[i].predict(Xtest)
            # try it on the validation test
            rmselist.append( sqrt(sum((Zpredicted-Ytest[:,i])**2)/len(Ytest[:,i]))  )
            RRMSElist.append( rmselist[i] / ( sum(Ytest[:,i])/len(Ytest[:,i])) )
            oob_scorelist.append( RFlist[i].oob_score_ )
            #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
            print(mon[i])
            print ('Test rmse: %.4f' % rmselist[i])
            print ('Test Rrmse: %.4f' % RRMSElist[i])
            print ('OOB score: %.4f' % oob_scorelist[i])

        return(RFlist,rmselist,RRMSElist,oob_scorelist)

    def PlotError_withNtrees(self, step, max_ntrees,plotTitle='RF Error evolution with n_trees',error='both'):
        '''
        Vizualise the evolution of test error and Out Of Bag error, as the number
        of trees increases, for different values of m (advised values from Breiman original paper).
        Plots the value of errors, at every ntrees step, until max_ntrees is reached.
        Can plot both Test rrmse and OOB error (error='both'), or only rrmse, or only oob
        '''
        X = self.Xlabel

        p =  X.shape[1]
        RF_list_m1,RF_list_m5,RF_list_m10,RF_list_m15 = [],[],[],[]
        for j in [i*step for i in range(1,int(max_ntrees/step))]:
            RF_list_m1.append(self.Fast_TrainAndTest( j,1))
            RF_list_m5.append(self.Fast_TrainAndTest( j, int(floor(p/6)) ))
            RF_list_m10.append(self.Fast_TrainAndTest( j,int(floor(p/3)) ))
            RF_list_m15.append(self.Fast_TrainAndTest( j,int(floor(2*p/3)) ))

        # PLOTS to compare multiple values for m, as a function of n_trees
        # I multiply by 100 to have errors in percent

        ''' NB: i dont create the figure and dont show() here to be able to use the function for months plots as well!'''
        if error=='both':
            #fig=plt.figure()
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,2 ],'b-',label=r'Test RRMSE, $m=1$')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], 1-array(RF_list_m1)[:,3 ], 'b--',label=r'OOB Error, $m=1$')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,2 ],'g-',label= r'Test RRMSE, $m= \left \lfloor \frac{d}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], 1-array(RF_list_m5)[:,3 ], 'g--',label=r'OOB Error, $m= \left \lfloor \frac{d}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,2],'r-',label= r'Test RRMSE, $m= \left \lfloor \frac{d}{3} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], 1-array(RF_list_m10)[:,3], 'r--',label= r'OOB Error, $m= \left \lfloor \frac{d}{3} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,2],'y-',label= r'Test RRMSE, $m= \left \lfloor \frac{2d}{3} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], 1-array(RF_list_m15)[:,3],'y--',label= r'OOB Error, $m= \left \lfloor \frac{2d}{3} \right \rfloor $')
            plt.xlabel('Number of trees $B$')
            plt.ylabel('Regression Error')
            #plt.axis((0,max_ntrees,0,1))
            #plt.legend()
            #plt.title(plotTitle)
            #plt.show()
        if error=='rrmse':
            #fig=plt.figure()
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,2] , 'b-',label=r'$m=1$')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,2] , 'g-',label= r'$m= \left \lfloor \frac{d}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,2], 'r-',label= r'$m= \left \lfloor \frac{d}{3} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,2], 'y-',label= r'$m= \left \lfloor \frac{2d}{3} \right \rfloor $')
            plt.xlabel('Number of trees $B$')
            plt.ylabel('Test NRMSE')
            #plt.axis((0,max_ntrees,0,1))
            #plt.legend()
            #plt.title(plotTitle)
            #plt.show()
        if error=='oob':
            #fig=plt.figure()
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,3] , 'b-',label=r'$m=1$')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,3] , 'g-',label=r'$m= \left \lfloor \frac{d}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,3], 'r-',label= r'$m= \left \lfloor \frac{d}{3} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,3],'y-',label= r'$m= \left \lfloor \frac{2d}{3} \right \rfloor $')
            plt.xlabel('Number of trees $B$')
            plt.ylabel('OOB Score')
            #plt.axis((0,max_ntrees,0,1))
            #plt.legend()
            #plt.title(plotTitle)
            #plt.show()
        #return(fig)



    def PlotError_withM(self,n_trees):
        '''
        Vizualises the evolution of errors, as m changes, with a predefined n_trees.
        The step for m is 1, and it tries all m until m = total number of features.
        '''
        p = self.Xlabel.shape[1]
        RF_list_m_100trees=[]
        for m in range(1,p+1):
            RF_list_m_100trees.append(self.Fast_TrainAndTest(n_trees,m)[1:])

        # PLOTS to compare errors as a function of m
        figure()
        plot(range(1,p+1), array(RF_list_m_100trees)[:,1], 'b-',label=r'Test Error')
        plot(range(1,p+1), 1 - array(RF_list_m_100trees)[:,3], 'b--',label=r'OOB Error')
        xlabel('m')
        ylabel('Regression Error')
        legend()
        title('RF error as a function of m, with n_trees= %.0f' % n_trees)
        show()

    def VariableImportance(self, n_trees, m, NbestFeatures, variables):
        '''
        Computes the variable importance for each feature, ranks the feature in ascending importance order,
        and selects a number NbetsFeatures of the "best" features, starting from the highest ranked one.
        variables:= the total list features.
        '''
        # average the variable importance over multiple RF training, to be more robust... #
        # the variable importance slightly change each time an rf is trained...
        # so we run it 100 times, and average the variable importance per variableFrame

        importances=[]
        for i in range(100):
            rf,rmse_rf,rrmse_rf,rf.oob_score_ = CustomRF(self.Xlabel,self.Ylabel).Fast_TrainAndTest(n_trees=n_trees,m= m)
            importances.append(rf.feature_importances_)
            print(i)
        AverageImportanceFrame = DataFrame( mean(array(importances),axis=0), columns=['importance_index'])
        AverageImportanceFrameSort = AverageImportanceFrame.sort(['importance_index'])

        fig=figure()
        yticks(range(70), [variables[:70][i] for i in list(AverageImportanceFrameSort.index)],  rotation=0)
        barh(range(70), array(AverageImportanceFrameSort.importance_index))
        xlabel('Variable importance from RF')
        title( ' '.join(['Variable importance from 100 RF trainings, for n_trees = %.0f' % n_trees,'and m = %.0f' % m])  )
        show()
        # selects the 20 first ranked features, by taking their indexes..
        indexes=list(AverageImportanceFrameSort.index[70-NbestFeatures:70])
        Best_features=[]
        for index in reversed(indexes):
            ''' we reverse the index list so that the best variables are ranked in decreasing importance'''
            Best_features.append(variables[index])

        return(Best_features,fig)

    def VariableImportance_general(self, n_trees, m, NbestFeatures, variables, FigVarNumber=None):
        '''
        not adpated to urban features anymore. Generalized for any variables, not including BFS and pixelFID.

        If we don't want to display all features but only a small number of them, specify it with FigVarNumber, is None we take by default all variables.
        '''
        # average the variable importance over multiple RF training, to be more robust... #
        # the variable importance slightly change each time an rf is trained...
        # so we run it 100 times, and average the variable importance per variableFrame

        importances=[]
        for i in range(100):
            rf,rmse_rf,rrmse_rf,rf.oob_score_ = CustomRF(self.Xlabel,self.Ylabel).Fast_TrainAndTest(n_trees=n_trees,m= m)
            importances.append(rf.feature_importances_)
            print(i)
        AverageImportanceFrame = pd.DataFrame( np.mean(np.array(importances),axis=0), columns=['importance_index'])
        AverageImportanceFrameSort = AverageImportanceFrame.sort_values(['importance_index'])


        # selects the best ranked features, by taking their indexes..
        indexes=list(AverageImportanceFrameSort.index[len(variables)-NbestFeatures:len(variables)])
        Best_features=[]
        for index in reversed(indexes):
            ''' we reverse the index list so that the best variables are ranked in decreasing importance'''
            Best_features.append(variables[index])


        # plot the VIs
        if FigVarNumber == None:

            fig=figure()
            yticks(range(len(variables)), [variables[i] for i in list(AverageImportanceFrameSort.index)], rotation=0)
            barh(range(len(variables)), array(AverageImportanceFrameSort.importance_index))
            xlabel('Variable importance from RF')
            title( ' '.join(['Variable importance from 100 RF trainings, for n_trees = %.0f' % n_trees,'and m = %.0f' % m])  )
            show()

        else:

            # selects the best
            indexes2=list(AverageImportanceFrameSort.index[len(variables)-FigVarNumber:len(variables)])
            Best_features_plot=[]
            for index in indexes2:
                Best_features_plot.append(variables[index])

            fig=figure()
            yticks(range(len(variables[:FigVarNumber])), Best_features_plot,  rotation=0)
            barh(range(len(variables[:FigVarNumber])), array(AverageImportanceFrameSort.importance_index)[-FigVarNumber:])
            xlabel('Variable importance from RF')
            title( ' '.join(['Variable importance from 100 RF trainings, for n_trees = %.0f' % n_trees,'and m = %.0f' % m])  )
            show()



        return(Best_features,fig)

    def TrainAndTest(self, n_trees, classification=False, AutomaticBounds="Yes", Xtr=None, Ytr=None, Xte=None, Yte=None):
        '''
        main two variables: n_trees, and m = max_features = number of features to sample from,
        m = integerPart(p/3) is adviced default value, where p is total number of features , but to test;
        then the minimum nodesize = min_samples_split = 5 is default value given by Breiman

        AutomaticBounds is a boolean parameter of the function: If "Yes", we automatically split between 75% training, 25% test,
        If "No", we need to specify Xtrn, Ytrn, Xtest, Ytest, if we need specific training and testing data
        '''

        if AutomaticBounds == "Yes":
            Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
            Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            # test set
            Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]


        if AutomaticBounds == "No":

            Xtrn, Ytrn = Xtr, Ytr
            # test set
            Xtest, Ytest = Xte, Yte


        # p is the total number of features
        p = self.Xlabel.shape[1]

        clf = RandomForestRegressor(n_estimators=n_trees)

        '''
        if p<6, the values advise by Breiman dont work (p/6 < 1), and we might as well test all m's,
        otherwise we test breimans values.
        '''
        if p > 6:
            param_grid = {"max_depth": [3,5, None],
                  #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                  "max_features": [1, int(floor(p/6)),int(floor(p/3)),int(floor(2*p/3))],
                  #"min_samples_split": [1, 3, 5, 7, 10],
                  "min_samples_leaf": [1,3,5]}
        else:
            param_grid = {"max_depth": [3, 5, None],
                  #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                  "max_features": [i for i in range(1,p+1)],
                  #"min_samples_split": [1, 3, 5, 7, 10],
                  "min_samples_leaf": [1,3,5]}

        # run grid search
        grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,cv=6)
        start = time()
        grid_search.fit(Xtrn, Ytrn)
        print("The best parameters are : ", grid_search.best_params_)
        #print("The best parameters are : ", clf.best_params_)
        # rf = RandomForestRegressor(n_estimators=n_trees,
        #                             oob_score=True,
        #                             max_depth=grid_search.best_params_['max_depth'],
        #                             max_features=grid_search.best_params_['max_features'],
        #                             min_samples_split=grid_search.best_params_['min_samples_split'],
        #                             min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
        rf = RandomForestRegressor(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2,
                                 max_depth = grid_search.best_params_['max_depth'],
                                 min_samples_leaf = grid_search.best_params_['min_samples_leaf'])
        rf.fit(Xtrn, Ytrn)
        Ypredicted = rf.predict(Xtest)

        Ypredicted_training = rf.predict(Xtrn)

        # try it on the validation test
        rmse_rf = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_rf = rmse_rf / ( sum(Ytest)/len(Ytest))

        # try it on the TRAINING test
        rmse_rf_train = sqrt(   sum((Ypredicted_training-Ytrn)**2) /len(Ytrn) )
        RRMSE_rf_train = rmse_rf_train / ( sum(Ytrn)/len(Ytrn))

        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test rmse: %.4f' % rmse_rf)
        print ('Test Rrmse: %.4f' % RRMSE_rf)
        print ('Training Rrmse: %.4f' % RRMSE_rf_train)
        print ('OOB score: %.4f' % rf.oob_score_)
        #Test rmse for SVR2D: 34995.2713
        return(rf,rmse_rf,RRMSE_rf, rf.oob_score_)



    def ExtractLeavesDistrib(self, model):
        '''
        returns the list of leaves dirtibution for each tree of a RF, in the form of a couple for each tree: (samplesnodesID list, list of lists of samples for all leaves ).

        example:

        ListOfTrees_leaves[0]

        ([3, 5, 6, 7, 9, 10],
 [array([ 13.51415344,  11.66080247,  12.76901902,  11.66080247]),
  array([ 11.80651322,  11.80651322,  12.26311111,  11.80651322,
          11.9822448 ,  11.9822448 ]),
  array([ 11.71032222,  11.6206834 ,  12.06218719]),
  array([ 11.0622807 ,  10.86969298,  11.00180556,  11.00180556,  11.02203013]),
  array([ 8.28771647,  8.63262821,  7.64741667,  8.63262821,  8.28771647,
          7.64741667]),
  array([ 5.06112103,  6.76233609,  6.76233609,  7.93032407,  6.76233609])])

        '''
        # ------------------------------------ Extract the distribution of each leaf, for all trees in the forest --------------------------------------------

        Boundtrain= int(round(0.75*len(self.Xlabel)))

        # look at the randomState for each tree and bootstrap the Ys in the same way
        ListOfTrees_leaves=[]

        for tree in model.estimators_:
            tree_randState = tree.random_state
            # set the seed to the same random state
            np.random.seed(tree_randState)
            choiceIndex=np.random.choice(range(Boundtrain),Boundtrain , replace=True)
            #bootstrap X and Y the same way
            Xtree = self.Xlabel[:Boundtrain][choiceIndex]
            Ytree = self.Ylabel[:Boundtrain][choiceIndex]
            # leaves for this tree when we put back Xtree
            leaves_column = DataFrame(tree.apply(Xtree), columns=['tree_i'])
            # corresponding Ytree together with leaves node ids
            # we concatenate this nodes matrix with the Y values of the samples, so that we know which label is in which leaf
            observations = concat([leaves_column,DataFrame(Ytree)],axis=1)
            # groupby leaf node id
            leaves_nodeList,leaves_samplesList =[],[]
            for leaf in observations.groupby(['tree_i']):
                leaf_id = leaf[0]
                # the samples in this leaf_id
                samples = array(leaf[1])[:,1]
                leaves_nodeList.append(leaf_id)
                leaves_samplesList.append(samples)

                ### look at things to verify it works well...
                #print("Leaf node id:")
                #print(leaf_id)
                #print("Samples in leaf:")
                #print(samples)
                #print("Number of unique samples:")
                #print(len(np.unique(samples)))
                #print("Value of leaf:")
                #print(np.mean(samples))
                #print('')
                ### VERIFIED with the visual of the tree! Ok.

            # for each tree, make a list of the couples: (leaves id list, list of distribution in each leaf)
            ListOfTrees_leaves.append((leaves_nodeList,leaves_samplesList))

        return(ListOfTrees_leaves)



    def QuantileRF_PIpred(self, model, XnewArray,ylim=None, ylim2=None , percentile=95,DoThePlot='False', ylabel=None, figName=None, resolutionCDF=0.5, color=None):
        '''
        Given a trained RF, a confidence value (default 95%), and an array of new points XnewArray, it returns PIs for each of the points, according to the QRF strategy, along with the mean value,
        in the form of an array where rows are list of triples: [down,mean,up].
        We add the possibility to plot the new points with the intervals.

        In case you want to set the ylim for visualisation purposes (for Ar for example!), ylim and ylim2 allow you to do it.
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))

        #xetract the distribution of samples in all leaves of the RF trees leaves
        ListOfTrees_leaves = CustomRF(self.Xlabel,self.Ylabel).ExtractLeavesDistrib(model)

        PI_test_list = []
        for New_x in XnewArray:

            # reshape for good format
            New_x = New_x.reshape(1, -1)

            # ------------ compute RF weights of samples for New_x -----------------------------------------
            # use apply function to see in which leaf of each tree it fell
            # I need to take [0] only when doing one x at a time because its an arry ine one element...
            RF_x_leaves= model.apply(New_x)[0]

            # compute weights of each yi (1 if in the leaf, else, 0, then normalize. Node that there are replacements, so the total contribution of a yi can be more than 1)
            wi_Trees_list = []  # the wis for each tree
            for i in range(len(ListOfTrees_leaves)):
                # leaf id where x fell in that tree
                leafId = RF_x_leaves[i]
                # look in the tree distribution which leaf id it is
                leafListIndex =  ListOfTrees_leaves[i][0].index(leafId)
                # take the samples distribution of that leaf
                leaf_samples = list(ListOfTrees_leaves[i][1][leafListIndex])

                # for each sample in leaf, count the number of times it appears in the leaf samples list, to compute the weight.
                # consider directy all the Yi values to have already the weights in order, to correspond with yi values.
                wi_unorm = [leaf_samples.count(s) for s in self.Ylabel[:Boundtrain]]
                # normalize to have the weights with the number of samples in the leaf of the tree
                wi = array(wi_unorm)/float(len(leaf_samples))
                wi_Trees_list.append(wi)

            # Now, we can compute the prediction of each tree by multiplying Weights*Y
            # to compute the weights of each training sample in the forest, simply average the weights through the trees
            RFsamplesWeights = mean(array(wi_Trees_list),axis=0)

            # ------------ compute conditional cdf of y given New_x -----------------------------------------8
            # compute the indicator function, for a fixed xx
            def cond_distrib(y):
                indic = array([1 if Yi<y else 0 for Yi in self.Ylabel[:Boundtrain]])
                wi_x = RFsamplesWeights
                prob = sum(wi_x*indic)
                return(prob)

                # compute cdf probs for points from 0 to 20 at a resolution of 0.001

            minT,maxT,resolution = min(self.Ylabel[:Boundtrain]) - 5  , max(self.Ylabel[:Boundtrain]) + 5  , resolutionCDF
            Tvalues = np.arange(minT,maxT,resolution)
            CDF_probs_new_x = array([cond_distrib(i) for i in Tvalues ])

            # ------------ compute 95% PI for New_x -----------------------------------------
            cdfFrame_x = DataFrame( np.vstack([Tvalues.T,CDF_probs_new_x.T ]).T , columns=['Tvalues','cdf'] )
            #compute quantiles by taking first y such that F>=alpha, corresponding to the inf...of course it depends on the resolution.

            alpha_2 = round((1 - 0.01*percentile)/2 , 3)
            q_up = array(cdfFrame_x[cdfFrame_x.cdf >=   1 - alpha_2])[0,0]
            q_down = array(cdfFrame_x[cdfFrame_x.cdf >=  alpha_2])[0,0]

            ## the RF mean estimate can be computed simply as RFWeights*Y (or of course by recalling model.predict)
            yPredMean = sum(RFsamplesWeights*self.Ylabel[:Boundtrain])

            # final PI with mean in between
            Interval_withMean = [q_down,yPredMean,q_up]

            # store it in list
            PI_test_list.append(Interval_withMean)

        # ------------ BONUS: possibility of plotting for new points  -----------------------------------------

        if DoThePlot=='True':
            '''
            down and up are the bottom and upper values, and not the differences between pred and down and up - thats what the errorbar
            function wants, the size of the errors. So we calculate differences.
            '''
            CI_errors=[np.array(PI_test_list)[:,1] - np.array(PI_test_list)[:,0] , np.array(PI_test_list)[:,2] - np.array(PI_test_list)[:,1]]

            fig = plt.figure()
            #plt.plot(range(N),np.array(down)[rand_indexes],color='skyblue' )
            #plt.plot(range(N),np.array(up)[rand_indexes],color='skyblue'  )
            plt.errorbar(range(len(XnewArray)),np.array(PI_test_list)[:,1], CI_errors,fmt='o',color=color)
            #plt.scatter(range(len(XnewArray)),np.array(PI_test_list)[:,0])
            #plt.scatter(range(len(XnewArray)),np.array(PI_test_list)[:,2])
            plt.xlabel('Predicted points')
            plt.ylabel(ylabel)
            plt.xlim([-1, len(XnewArray)])
            plt.ylim([ylim, ylim2])
            plt.show()

            fig.savefig(''.join([figName,'.pdf']), bbox_inches='tight', dpi=400)

        return(np.array(PI_test_list))





    def QRF_CV(self,model,figName,ylabel,xy,xytext,x_axis, x_axis2, y_axis, y_axis2,pointSize,lineWidth,BeginTestRatio=0.75,EndTestRatio=1  ,percentile=95,resolutionCDF=0.5,PutLegend="True",legLoc = 'upper right'):

        '''
        applies QuantileRF_PIpred to the test set, to cross validate the intervals with the real labels.
        Computes the intervals, and plot the intervals, together with observed values, and the actual percentage of observed values within intervals.

        Prints the actual percentage of observed values within intervals.
        Saves the figure of the intervals plots, under the name figName.

        to be in the function, I needed to let the plot parameters be variables of the function...like the axis x and y, and the location of the annotation...

        EndTestRatio is defined in case the test case is very big and you might not want to plot all test samples. EndTestRatio = 0.85 is for example to plot the part of the test set corresponding to 10 % of the total labeled set.

        '''

        Boundtrain= int(round(0.75*len(self.Xlabel)))

        BoundBeginTest= int(round(BeginTestRatio*len(self.Xlabel)))

        BoundEndTest= int(round(EndTestRatio*len(self.Xlabel)))

        PI_array = CustomRF(self.Xlabel,self.Ylabel).QuantileRF_PIpred(model, self.Xlabel[BoundBeginTest:BoundEndTest],percentile=percentile,resolutionCDF=resolutionCDF)

        ytest = self.Ylabel[BoundBeginTest:BoundEndTest]
        err_downlist,mean_list,err_uplist = PI_array[:,0],PI_array[:,1],PI_array[:,2]

        # count the percentage of observed within the intervals
        WithinCI=0
        for i in range(len(ytest)):
            if err_downlist[i] <= ytest[i] <= err_uplist[i]:
                WithinCI = WithinCI + 1
        FreqWithinCI = WithinCI/float(len(ytest))

        fig = plt.figure()
        plt.scatter(range(len(ytest)) , ytest, c='blue',s=pointSize, label='Observed')
        plt.scatter(range(len(ytest)) ,mean_list, c='red',s=pointSize,label='Predicted')
        plt.plot(range(len(ytest)) ,err_downlist, c='red',linewidth=lineWidth,label='95% PI')
        plt.plot(range(len(ytest)) ,err_uplist, linewidth=lineWidth, c='red')
        if PutLegend=="True":
            plt.legend(loc=legLoc)
        plt.xlabel('Test points')
        plt.ylabel(ylabel)
        plt.annotate('Test confidence : %.2f' % FreqWithinCI, xy=xy, xytext=xytext)
        plt.axis([x_axis, x_axis2, y_axis, y_axis2])
        plt.show()

        print ('Percentage of observed values within the PI: %.4f' % FreqWithinCI)
        fig.savefig( ''.join([figName,'.pdf']),bbox_inches='tight',dpi=400)




    def Monthly_TrainAndTest(self, n_trees, Ymonths):
        '''
        Train and test in the case of monthly data.
        Ymonths is the matrix of monthly output values, (N by 12 matrix).
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],Ymonths[:Boundtrain]
            #validation set
        Xtest, Ytest = self.Xlabel[Boundtrain:],Ymonths[Boundtrain:]

        RFlist,rmselist,RRMSElist,oob_scorelist = [],[],[],[]
        mon = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        # p is the total number of features
        p = self.Xlabel.shape[1]

        #################################
        for i in range(len(mon)):

            '''
            if p<6, the values advise by Breiman dont work (p/6 < 1), and we might as well test all m's,
            otherwise we test breimans values.
            '''
            if p > 6:
                param_grid = {#"max_depth": [3, None],
                      #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                      "max_features": [1, int(floor(p/6)),int(floor(p/3)),int(floor(2*p/3))]}
                      #"min_samples_split": [1, 3, 5, 7, 10],
                      #"min_samples_leaf": [1, 3, 5, 7, 10]}
            else:
                param_grid = {#"max_depth": [3, None],
                      #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                      "max_features": [j for j in range(1,p+1)]}
                      #"min_samples_split": [1, 3, 5, 7, 10],
                      #"min_samples_leaf": [1, 3, 5, 7, 10]}
            # run grid search
            clf = RandomForestRegressor(n_estimators=n_trees)
            grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,cv=6)
            #start = time()
            grid_search.fit(Xtrn, Ytrn[:,i])
            print("The best parameters are : ", grid_search.best_params_)
            #print("The best parameters are : ", clf.best_params_)
            # rf = RandomForestRegressor(n_estimators=n_trees,
            #                             oob_score=True,
            #                             max_depth=grid_search.best_params_['max_depth'],
            #                             max_features=grid_search.best_params_['max_features'],
            #                             min_samples_split=grid_search.best_params_['min_samples_split'],
            #                             min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
            RFlist.append( RandomForestRegressor(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2))
            RFlist[i].fit(Xtrn, Ytrn[:,i])
            Zpredicted=RFlist[i].predict(Xtest)
            # try it on the validation test
            rmselist.append( sqrt(sum((Zpredicted-Ytest[:,i])**2)/len(Ytest[:,i]))  )
            RRMSElist.append( rmselist[i] / ( sum(Ytest[:,i])/len(Ytest[:,i])) )
            oob_scorelist.append( RFlist[i].oob_score_ )
            #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
            print(mon[i])
            print ('Test rmse: %.4f' % rmselist[i])
            print ('Test Rrmse: %.4f' % RRMSElist[i])
            print ('OOB score: %.4f' % oob_scorelist[i])

        return(RFlist,rmselist,RRMSElist,oob_scorelist)

    def pred_ints(self,model, Z, percentile=95):
        '''
        calculates bottom and up errors for CI, for a Xtest Z.
        Computes the predictions of points for each boostrap tree, and then calculate percentiles on the trees prediction for each point.
        '''
        trees_predictions = model.estimators_[0].predict(Z)
        for i in range(1,len(model.estimators_)):
            trees_predictions = np.vstack([trees_predictions,model.estimators_[i].predict(Z)  ])

        err_down = []
        err_up = []
        for j in range(trees_predictions.shape[1]):
            err_down.append(np.percentile(trees_predictions[:,j],(100 - percentile) / 2.))
            err_up.append(np.percentile(trees_predictions[:,j],100 - (100 - percentile) / 2.))
        return err_down, err_up

    def Fast_TrainAndTest_withCIs(self, n_trees, m, percentile):
        '''
        Here we simply expland the trees fully so that each leaf has exactly one sample value, so that each leaf has an original sample value
        and not means of original values.
        (as Breiman suggested and the blog http://blog.datadive.net/prediction-intervals-for-random-forests/)
        to be able to derive CI without the original RF quantile regression idea.
        Finally, I ONLY CHANGE min_samples_leaf = 1
        '''
        Boundtrain= round(0.75*len(self.Xlabel))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            # test set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        rf = RandomForestRegressor(n_estimators=n_trees,
                                     oob_score=True,
                                     max_depth= None,
                                     max_features= m ,
                                     min_samples_split=2,
                                    min_samples_leaf = 1)
        rf.fit(Xtrn, Ytrn)
        Ypredicted = rf.predict(Xtest)
        # try it on the validation test
        rmse_rf = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_rf = rmse_rf / ( sum(Ytest)/len(Ytest))
        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)

        ##### calculates CIs for each prediction !! ##########
        err_down, err_up = CustomRF(self.Xlabel,self.Ylabel).pred_ints(rf,Xtest,percentile)

        # print errors
        print ('Test rmse: %.4f' % rmse_rf)
        print ('Test Rrmse: %.4f' % RRMSE_rf)
        print ('OOB score: %.4f' % rf.oob_score_)
        #Test rmse for SVR2D: 34995.2713
        return(rf,rmse_rf,RRMSE_rf, rf.oob_score_,err_down,err_up,Ytest,Ypredicted )

    def TrainAndTest_withCIs(self, n_trees, percentile):
        '''
        Here we simply expland the trees fully so that each leaf has exactly one sample value, so that each leaf has an original sample value
        and not means of original values.
        (as Breiman suggested and the blog http://blog.datadive.net/prediction-intervals-for-random-forests/)
        to be able to derive CI without the original RF quantile regression idea.
        Finally, I ONLY CHANGE min_samples_leaf = 1
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            # test set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        p = self.Xlabel.shape[1]

        '''
        if p<6, the values advise by Breiman dont work (p/6 < 1), and we might as well test all m's,
        otherwise we test breimans values.
        '''
        if p > 7:
            param_grid = {#"max_depth": [3, None],
                  #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                  "max_features": [1, int(floor(p/6)),int(floor(p/3)),int(floor(2*p/3))]}
                  #"min_samples_split": [1, 3, 5, 7, 10],
                  #"min_samples_leaf": [1, 3, 5, 7, 10]}
        else:
            param_grid = {#"max_depth": [3, None],
                  #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                  "max_features": [i for i in range(1,p+1)]}
                  #"min_samples_split": [1, 3, 5, 7, 10],
                  #"min_samples_leaf": [1, 3, 5, 7, 10]}

        # run grid search
        clf = RandomForestRegressor(n_estimators=n_trees)
        grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,cv=6)
        grid_search.fit(Xtrn, Ytrn)
        print("The best parameters are : ", grid_search.best_params_)
        #print("The best parameters are : ", clf.best_params_)
        # rf = RandomForestRegressor(n_estimators=n_trees,
        #                             oob_score=True,
        #                             max_depth=grid_search.best_params_['max_depth'],
        #                             max_features=grid_search.best_params_['max_features'],
        #                             min_samples_split=grid_search.best_params_['min_samples_split'],
        #                             min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
        rf = RandomForestRegressor(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2,
                                 min_samples_leaf = 1)
        rf.fit(Xtrn, Ytrn)
        Ypredicted = rf.predict(Xtest)
        # try it on the validation test
        rmse_rf = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_rf = rmse_rf / ( sum(Ytest)/len(Ytest))
        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)

        ##### calculates CIs for each prediction !! ##########
        err_down, err_up = CustomRF(self.Xlabel,self.Ylabel).pred_ints(rf,Xtest,percentile)

        # print errors
        print ('Test rmse: %.4f' % rmse_rf)
        print ('Test Rrmse: %.4f' % RRMSE_rf)
        print ('OOB score: %.4f' % rf.oob_score_)
        #Test rmse for SVR2D: 34995.2713
        return(rf,rmse_rf,RRMSE_rf, rf.oob_score_,err_down,err_up,Ytest,Ypredicted )

#import numpy as np
#def shuffle_in_unison(a, b):
#    rng_state = np.random.get_state()
#    np.random.shuffle(a)
#    np.random.set_state(rng_state)
#    np.random.shuffle(b)



    def Fast_TrainAndTest_2Averaging(self, n_trees1, m1, n_trees2, m2):
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            # test set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        rf1 = RandomForestRegressor(n_estimators=n_trees1,
                                     oob_score=True,
                                     max_depth= None,
                                     max_features= m1 ,
                                     min_samples_split=2,
                                    min_samples_leaf = 3)
        rf1.fit(Xtrn, Ytrn)
        rf2= RandomForestRegressor(n_estimators=n_trees2,
                                     oob_score=True,
                                     max_depth= None,
                                     max_features= m2 ,
                                     min_samples_split=2,
                                    min_samples_leaf = 3)
        rf2.fit(Xtrn, Ytrn)
        #rf2 = ExtraTreesRegressor(n_estimators=n_trees2, max_features=m2,bootstrap=True,oob_score=True)
        #rf2.fit(Xtrn, Ytrn)


        Ypredicted1 =  rf1.predict(Xtest)
        Ypredicted2 = rf2.predict(Xtest)

        Ypredicted =  ( rf1.predict(Xtest) + rf2.predict(Xtest))/2
        # try it on the validation test
        rmse_rf = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_rf = rmse_rf / ( sum(Ytest)/len(Ytest))

        rmse_rf1 = sqrt(   sum((Ypredicted1-Ytest)**2) /len(Ytest) )
        RRMSE_rf1 = rmse_rf1 / ( sum(Ytest)/len(Ytest))
        rmse_rf2 = sqrt(   sum((Ypredicted2-Ytest)**2) /len(Ytest) )
        RRMSE_rf2 = rmse_rf2 / ( sum(Ytest)/len(Ytest))
        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test rmse 1: %.4f' % rmse_rf1)
        print ('Test Rrmse 1: %.4f' % RRMSE_rf1)
        print ('OOB score 1: %.4f' % rf1.oob_score_)
        print ('Test rmse 2: %.4f' % rmse_rf2)
        print ('Test Rrmse 2: %.4f' % RRMSE_rf2)
        print ('OOB score 2: %.4f' % rf2.oob_score_)
        print('')
        print ('Test rmse Averaging: %.4f' % rmse_rf)
        print ('Test Rrmse Averaging: %.4f' % RRMSE_rf)

        print('salut')
        #Test rmse for SVR2D: 34995.2713
        return(rf1, rf2, rmse_rf,RRMSE_rf,rf1.oob_score_,rf2.oob_score_)





    def Fast_TrainAndTest_RandomFeaturesAveraging(self, n_trees, m, K, F):

        '''
        K is the number of uncorrelated regressors we want to build using F features randomly chosen from
        the d features (F must be smaller than d of course!!!!)
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
            # test set
        Xtest, Ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]

        '''
        Create K lists of 10 random indexes, to be able to create 5 different regressors
        using only the 10 corresponding features to train out of all features.
        '''

        rand_indexes_list=[]
        for lol in range(K):
            rand_indexes=[]
            for lolo in range(F):
                rand_indexes.append(random.randint(0,69))
            rand_indexes_list.append(rand_indexes)
        rand_indexes_array = np.array(rand_indexes_list)

        # creates K random forests with the same parameters, but that will be fit in 5 different
        # samples using different features of the data
        rfs,Ypredictions = [],[]

        for i in range(K):
            rfs.append(RandomForestRegressor(n_estimators=n_trees,
                                     oob_score=True,
                                     max_depth= None,
                                     max_features= m ,
                                     min_samples_split=2,
                                    min_samples_leaf = 1))
            rfs[i].fit(Xtrn[:,rand_indexes_array[i,:]], Ytrn)
            Ypredictions.append(rfs[i].predict(Xtest[:,rand_indexes_array[i,:]]))


        Ypredicted = np.mean(Ypredictions,0)
        # try it on the validation test
        rmse_rf = sqrt(   sum((Ypredicted-Ytest)**2) /len(Ytest) )
        RRMSE_rf = rmse_rf / ( sum(Ytest)/len(Ytest))

        #rmse_rf1 = sqrt(   sum((Ypredicted1-Ytest)**2) /len(Ytest) )
        #RRMSE_rf1 = rmse_rf1 / ( sum(Ytest)/len(Ytest))
        #rmse_rf2 = sqrt(   sum((Ypredicted2-Ytest)**2) /len(Ytest) )
        #RRMSE_rf2 = rmse_rf2 / ( sum(Ytest)/len(Ytest))
        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        #print ('Test rmse 1: %.4f' % rmse_rf1)
        #print ('Test Rrmse 1: %.4f' % RRMSE_rf1)
        #print ('OOB error 1: %.4f' % rf1.oob_score_)
        #print ('Test rmse 2: %.4f' % rmse_rf2)
        #print ('Test Rrmse 2: %.4f' % RRMSE_rf2)
        #print ('OOB error 2: %.4f' % rf2.oob_score_)
        #print('')
        print ('Test rmse Averaging: %.4f' % rmse_rf)
        print ('Test Rrmse Averaging: %.4f' % RRMSE_rf)

        print('salut')
        #Test rmse for SVR2D: 34995.2713
        return( rmse_rf,RRMSE_rf)



    def TrainAndTest_class(self, n_trees):
        '''
        main two variables: n_trees, and m = max_features = number of features to sample from,
        m = integerPart(sqrt(d)) is adviced default value, where p is total number of features , but to test;
        then the minimum nodesize = min_samples_split = 5 is default value given by Breiman
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:int(Boundtrain)],self.Ylabel[:int(Boundtrain)]
            # test set
        Xtest, Ytest = self.Xlabel[int(Boundtrain):],self.Ylabel[int(Boundtrain):]
        # p is the total number of features
        p = self.Xlabel.shape[1]

        clf = RandomForestClassifier(n_estimators=n_trees)

        '''
        if p<6, the values advise by Breiman dont work (p/6 < 1), and we might as well test all m's,
        otherwise we test breimans values.
        '''

        param_grid = {#"max_depth": [3, None],
                  #"max_features": [1, 5, 10, 15, 20, 25 ,30, 35, 40],
                  "max_features": [1, int(np.sqrt(p/2)),int(np.sqrt(p)),int(np.sqrt(2*p)) ],
                  #"min_samples_split": [1, 3, 5, 7, 10],
                  "min_samples_leaf": [1, 3, 5, 7, 10]}

        # run grid search
        grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,cv=6)
        start = time()
        grid_search.fit(Xtrn, Ytrn)
        print("The best parameters are : ", grid_search.best_params_)
        #print("The best parameters are : ", clf.best_params_)
        # rf = RandomForestRegressor(n_estimators=n_trees,
        #                             oob_score=True,
        #                             max_depth=grid_search.best_params_['max_depth'],
        #                             max_features=grid_search.best_params_['max_features'],
        #                             min_samples_split=grid_search.best_params_['min_samples_split'],
        #                             min_samples_leaf=grid_search.best_params_['min_samples_leaf'])
        rf = RandomForestClassifier(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2,
                                 min_samples_leaf = grid_search.best_params_['min_samples_leaf'])
        rf.fit(Xtrn, Ytrn)
        Ypredicted = rf.predict(Xtest)

        Ypredicted_training = rf.predict(Xtrn)

        accuracy = float(sum(Ypredicted==Ytest))/float(len(Ytest))
        accuracy_train = float(sum(Ypredicted_training==Ytrn))/float(len(Ytrn))

        confMat = confusion_matrix(Ytest,Ypredicted)
        AccPerClass = diag(confMat)/(confMat.sum(axis=1))
        AccPerClassRound = array([round(el,2) for el in  AccPerClass*100])
        #confMatWithAcc = vstack([ array(confMat).T,AccPerClassRound.T ]).T

        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test accuracy for RF: %.4f' % accuracy)
        print ('Train accuracy for RF: %.4f' % accuracy_train)
        print ('OOB score: %.4f' % rf.oob_score_)
        # the values in the last column of the matrix are individual class accuracies, in percentage (10 for 10%)
        print(confMat)
        print(AccPerClassRound)
        #Test rmse for SVR2D: 34995.2713
        return(rf,accuracy, rf.oob_score_)


    def Fast_TrainAndTest_class(self, n_trees,m):
        '''
        main two variables: n_trees, and m = max_features = number of features to sample from,
        m = integerPart(p/3) is adviced default value, where p is total number of features , but to test;
        then the minimum nodesize = min_samples_split = 5 is default value given by Breiman
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))
            # train set
        Xtrn, Ytrn = self.Xlabel[:int(Boundtrain)],self.Ylabel[:int(Boundtrain)]
            # test set
        Xtest, Ytest = self.Xlabel[int(Boundtrain):],self.Ylabel[int(Boundtrain):]
        # p is the total number of features
        clf = RandomForestClassifier(n_estimators=n_trees)


        rf = RandomForestClassifier(n_estimators=n_trees,
                                oob_score=True,
                                max_features=m,
                                 min_samples_split = 2,
                                 min_samples_leaf = 1)
        rf.fit(Xtrn, Ytrn)
        Ypredicted = rf.predict(Xtest)

        Ypredicted_training = rf.predict(Xtrn)

        accuracy = float(sum(Ypredicted==Ytest))/float(len(Ytest))
        accuracy_train = float(sum(Ypredicted_training==Ytrn))/float(len(Ytrn))

        confMat = confusion_matrix(Ytest,Ypredicted)
        AccPerClass = diag(confMat)/(confMat.sum(axis=1))
        AccPerClassRound = array([round(el,2) for el in AccPerClass*100])
        #confMatWithAcc = vstack([ confMat.T,AccPerClassRound.T ]).T


        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        print ('Test accuracy for RF: %.4f' % accuracy)
        print ('Train accuracy for RF: %.4f' % accuracy_train)
        print ('OOB error: %.4f' % rf.oob_score_)
        # the values in the last column of the matrix are individual class accuracies, in percentage (10 for 10%)
        print(confMat)
        print(AccPerClassRound)
        #Test rmse for SVR2D: 34995.2713
        return(rf,accuracy, rf.oob_score_)

    def VariableImportance_general_class(self, n_trees, m, NbestFeatures, variables):
        '''
        not adpated to urban features anymore. Generalized for any variables, not including BFS and pixelFID
        '''
        # average the variable importance over multiple RF training, to be more robust... #
        # the variable importance slightly change each time an rf is trained...
        # so we run it 100 times, and average the variable importance per variableFrame

        importances=[]
        for i in range(100):
            rf,accuracy,rf_oob_score_ = CustomRF(self.Xlabel,self.Ylabel).Fast_TrainAndTest_class(n_trees=n_trees,m= m)
            importances.append(rf.feature_importances_)
            print(i)
        AverageImportanceFrame = pd.DataFrame( np.mean(np.array(importances),axis=0), columns=['importance_index'])
        AverageImportanceFrameSort = AverageImportanceFrame.sort_values(['importance_index'])

        fig=figure()
        yticks(range(len(variables)), [variables[i] for i in list(AverageImportanceFrameSort.index)],  rotation=0)
        barh(range(len(variables)), array(AverageImportanceFrameSort.importance_index))
        xlabel('Variable importance from RF')
        title( ' '.join(['Variable importance from 100 RF trainings, for n_trees = %.0f' % n_trees,'and m = %.0f' % m])  )
        show()
        # selects the 20 first ranked features, by taking their indexes..
        indexes=list(AverageImportanceFrameSort.index[len(variables)-NbestFeatures:len(variables)])
        Best_features=[]
        for index in reversed(indexes):
            ''' we reverse the index list so that the best variables are ranked in decreasing importance'''
            Best_features.append(variables[index])

        return(Best_features,fig)


    def PlotError_withNtrees_class(self, step, max_ntrees,plotTitle='RF Accuracy evolution with n_trees',error='both'):
        '''
        Vizualise the evolution of test error and Out Of Bag error, as the number
        of trees increases, for different values of m (advised values from Breiman original paper).
        Plots the value of errors, at every ntrees step, until max_ntrees is reached.
        Can plot both Test accuracy and OOB error (error='both'), or only 'acc', or only 'oob'
        '''
        X = self.Xlabel

        p =  X.shape[1]
        RF_list_m1,RF_list_m5,RF_list_m10,RF_list_m15 = [],[],[],[]
        for j in [i*step for i in range(1,int(max_ntrees/step))]:
            RF_list_m1.append(self.Fast_TrainAndTest_class( j,1))
            #RF_list_m5.append(self.Fast_TrainAndTest_class( j, int(floor(sqrt(p))) ))
            RF_list_m10.append(self.Fast_TrainAndTest_class( j,int(floor(sqrt(p))) ))
            RF_list_m15.append(self.Fast_TrainAndTest_class( j,int(floor(2*sqrt(p))) ))

        # PLOTS to compare multiple values for m, as a function of n_trees
        # I multiply by 100 to have errors in percent

        ''' NB: i dont create the figure and dont show() here to be able to use the function for months plots as well!'''
        if error=='both':
            fig=plt.figure()
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,1]*100, 'b-',label=r'Test Acc, $m=1$')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,2]*100, 'b--',label=r'OOB Error, $m=1$')
            #plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,1]*100, 'g-',label= r'Test Acc, $m= \left \lfloor \frac{p}{6} \right \rfloor $')
            #plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,2]*100, 'g--',label=r'OOB Error, $m= \left \lfloor \frac{p}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,1]*100, 'r-',label= r'Test Acc, $m= \left \lfloor \sqrt{p} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,2]*100, 'r--',label= r'OOB Error, $m= \left \lfloor \sqrt{p} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,1]*100, 'y-',label= r'Test Acc, $m= \left \lfloor 2\sqrt{p} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,2]*100,'y--',label= r'OOB Error, $m= \left \lfloor 2\sqrt{p} \right \rfloor $')
            plt.xlabel('Number of trees')
            plt.ylabel('Accuracy (percent)')
            plt.legend()
            plt.title(plotTitle)
            plt.show()
        if error=='acc':
            fig=plt.figure()
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,1]*100, 'b-',label=r'Test Acc, $m=1$')
            #plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,1]*100, 'g-',label= r'Test Acc, $m= \left \lfloor \frac{p}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,1]*100, 'r-',label= r'Test Acc, $m= \left \lfloor \sqrt{p} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,1]*100, 'y-',label= r'Test Acc, $m= \left \lfloor 2\sqrt{p} \right \rfloor $')
            plt.xlabel('Number of trees')
            plt.ylabel('Accuracy (percent)')
            plt.legend()
            plt.title(plotTitle)
            plt.show()
        if error=='oob':
            fig=plt.figure()
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m1)[:,2]*100, 'b--',label=r'OOB Error, $m=1$')
            #plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m5)[:,2]*100, 'g--',label=r'OOB Error, $m= \left \lfloor \frac{p}{6} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m10)[:,2]*100, 'r--',label= r'OOB Error, $m= \left \lfloor \sqrt{p} \right \rfloor $')
            plt.plot([i*step for i in range(1,int(max_ntrees/step))], array(RF_list_m15)[:,2]*100,'y--',label= r'OOB Error, $m= \left \lfloor 2\sqrt{p} \right \rfloor $')
            plt.xlabel('Number of trees')
            plt.ylabel('Accuracy (percent)')
            plt.legend()
            plt.title(plotTitle)
            plt.show()


        return(fig,RF_list_m1,RF_list_m5,RF_list_m10,RF_list_m15)




    def Stacking_class(self, n_trees):
        '''
        main two variables: n_trees, and m = max_features = number of features to sample from,
        m = integerPart(p/3) is adviced default value, where p is total number of features , but to test;
        then the minimum nodesize = min_samples_split = 5 is default value given by Breiman
        '''
        Boundtrain= int(round(0.75*len(self.Xlabel)))

        # train set
        Xtrn, ytrn = self.Xlabel[:Boundtrain],self.Ylabel[:Boundtrain]
        # test set
        Xtest, ytest = self.Xlabel[Boundtrain:],self.Ylabel[Boundtrain:]
        # p is the total number of features
        p = self.Xlabel.shape[1]


        #-------------- Split train sets in Train set a and b  ----------------

        BoundSplit = int(round(0.5*len(Xtrn)))

        Xtrn_a,ytrn_a = Xtrn[:BoundSplit],ytrn[:BoundSplit]
        Xtrn_b,ytrn_b = Xtrn[BoundSplit:],ytrn[BoundSplit:]

        #--------------- train model a ------------------------
        clf_a = RandomForestClassifier(n_estimators=n_trees)
        if p > 6:
            param_grid = {"max_features": [1, int(floor(p/6)),int(floor(p/3)),int(floor(2*p/3))]}
        else:
            param_grid = {"max_features": [i for i in range(1,p+1)]}
        grid_search = GridSearchCV(clf_a, param_grid=param_grid,n_jobs=-1,cv=6)
        start = time()
        grid_search.fit(Xtrn_a, ytrn_a)
        print("training model with train a... best params: ", grid_search.best_params_)

        rf_a = RandomForestClassifier(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2,
                                 min_samples_leaf = 1)
        rf_a.fit(Xtrn_a, ytrn_a)

        #------------------ train model b ------------------------
        clf_b = RandomForestClassifier(n_estimators=n_trees)
        if p > 6:
            param_grid = {"max_features": [1, int(floor(p/6)),int(floor(p/3)),int(floor(2*p/3))]}
        else:
            param_grid = {"max_features": [i for i in range(1,p+1)]}
        grid_search = GridSearchCV(clf_b, param_grid=param_grid,n_jobs=-1,cv=6)
        start = time()
        grid_search.fit(Xtrn_b, ytrn_b)
        print("training model with train b... best params : ", grid_search.best_params_)

        rf_b = RandomForestClassifier(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2,
                                 min_samples_leaf = 1)
        rf_b.fit(Xtrn_b, ytrn_b)

        #------------------ train model on full training set ------------------------
        clf = RandomForestClassifier(n_estimators=n_trees)
        if p > 6:
            param_grid = {"max_features": [1, int(floor(p/6)),int(floor(p/3)),int(floor(2*p/3))]}
        else:
            param_grid = {"max_features": [i for i in range(1,p+1)]}
        grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,cv=6)
        start = time()
        grid_search.fit(Xtrn, ytrn)
        print("training model with full training set... best params: ", grid_search.best_params_)

        rf = RandomForestClassifier(n_estimators=n_trees,
                                oob_score=True,
                                max_features=grid_search.best_params_['max_features'],
                                 min_samples_split = 2,
                                min_samples_leaf = 1)
        rf.fit(Xtrn, ytrn)

        #----------- create predictions -------------------

        NewXtrn_a = rf_b.predict_proba(Xtrn_a)
        NewXtrn_b = rf_a.predict_proba(Xtrn_b)
        NewXtest = rf.predict_proba(Xtest)

        print(NewXtrn_a.shape)
        print(NewXtrn_b.shape)
        print(NewXtest.shape)

        NewX = vstack([NewXtrn_a,NewXtrn_b,NewXtest])
        rfStack,acc,StackOOB = CustomRF(NewX,self.Ylabel).TrainAndTest_class(n_trees)

        #---------- Train second stage model --------------


        #------------
        #accuracy = float(sum(Ypredicted==Ytest))/float(len(Ytest))

        #print ('CV rmse for SVR2D: %.4f' % rmseIValid)
        #print ('Test accuracy for RF: %.4f' % accuracy)
        #print ('OOB error: %.4f' % rf.oob_score_)
        #print(confusion_matrix(Ytest,Ypredicted))
        #Test rmse for SVR2D: 34995.2713
        return(rfStack,acc,StackOOB)
