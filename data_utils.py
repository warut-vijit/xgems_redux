import numpy as np
import scipy
import os, sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

BATCH_SIZE=1
class DefaultCreditDataSampler(object):
    def __init__(self,one_hot=True):
        self.n_labels = 2
        self.xtrain=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/xtrain.csv',header=None))
        self.xtrain_hivae=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/xtrain_hivae.csv',header=None))
        self.xtest=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/xtest.csv',header=None))
        self.xtest_hivae=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/xtest_hivae.csv',header=None))
        self.ytrain=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/ytrain.csv',header=None))
        print(np.where(self.ytrain==1)[0].shape,self.ytrain.shape)
        self.ytest=np.asarray(pd.read_csv('/h/shalmali/data/defaultCredit/ytest.csv',header=None))
        self.scaler=MinMaxScaler()
        self.scaler.fit(self.xtrain)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.ytrain=self.enc.fit_transform(self.ytrain).toarray()
        self.ytest = self.enc.transform(self.ytest).toarray()
        #print(self.ytest.shape,type(self.ytest))
        self.xtrain = self.scaler.transform(self.xtrain)
        self.xtest = self.scaler.transform(self.xtest)
        self.n_samples,self.n_features=self.xtrain.shape
        self.n_samples_test = self.xtest.shape[0]
        self.shape=[self.n_features]

    def __call__(self, batch_size,batch_index,normalized=True):
        idx_start=batch_index*batch_size
        idx_end=(batch_index+1)*batch_size
        return self.xtrain[idx_start:idx_end,:],self.ytrain[idx_start:idx_end,:]

    def get_test(self):
        return self.xtest, self.ytest

    def get_test_i(self,i):
        return np.reshape(self.xtest[i,:],[1,-1]),np.reshape(self.ytest[i,:],[1,-1])


class IHDPDataSampler(object):
    def __init__(self, replications=10):
        self.path='IHDP/csv'
        self.replications=replications
        # which features are binary
        self.binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        # which features are continuous
        self.contfeats = [i for i in range(25) if i not in self.binfeats]
        data = []
        for i in range(self.replications):
            replica_path = self.path + '/ihdp_npci_' + str(i + 1) + '.csv'
            data.append(np.loadtxt(replica_path, delimiter=','))
        data = np.vstack(data)
        self.n_labels = 1
        t, y = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis]
        #x = np.hstack([t, data[:, 5:]]) # fields 3 and 4 are mus and not used.
        x = data[:, 5:]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size=0.1)

        print(self.xtrain.shape, self.ytrain.shape, self.xtest.shape, self.ytest.shape)
        self.n_samples,self.n_features=self.xtrain.shape
        self.n_samples_test = self.xtest.shape[0]
        self.shape=[self.n_features]

    def __call__(self, batch_size,batch_index,normalized=True):
        idx_start=batch_index*batch_size
        idx_end=(batch_index+1)*batch_size
        return self.xtrain[idx_start:idx_end,:],self.ytrain[idx_start:idx_end,:]

    def get_test(self):
        return self.xtest, self.ytest

    def get_test_i(self,i):
        return np.reshape(self.xtest[i,:],[1,-1]),np.reshape(self.ytest[i,:],[1,-1])


class TwinsDataSampler(object):
    def __init__(self, p_heavier=0.5):
        self.n_labels = 2
        x_raw = pd.read_csv('twins/twin_pairs_X_3years_samesex.csv')
        t_raw = pd.read_csv('twins/twin_pairs_T_3years_samesex.csv')
        y_raw = pd.read_csv('twins/twin_pairs_Y_3years_samesex.csv')
        # select the 46 columns from CEVAE paper
        fields = ['adequacy', 'alcohol', 'anemia', 'birattnd', 'brstate', 'brstate_reg', 'cardiac', 'chyper', 'cigar6', 'crace', 'csex', 'dfageq', 'diabetes', 'dlivord_min', 'dmar', 'drink5', 'dtotord_min', 'eclamp', 'feduc6', 'frace', 'gestat10', 'hemo', 'herpes', 'hydra', 'incervix', 'lung', 'mager8', 'meduc6', 'mplbir', 'mplbir_reg', 'mpre5', 'mrace', 'nprevistq', 'orfath', 'ormoth', 'othermr', 'phyper', 'pldel', 'pre4000', 'preterm', 'renal', 'rh', 'stoccfipb', 'stoccfipb_reg', 'tobacco', 'uterine']
        x_raw = x_raw[fields]
        # select rows where both twins <2kg at birth
        low_weight = (t_raw["dbirwt_0"]<2000) & (t_raw["dbirwt_1"]<2000)
        x = np.asarray(x_raw[low_weight])
        t = np.asarray(t_raw[low_weight])
        y = np.asarray(y_raw[low_weight])

        # select only one of the twins
        xlight, xheavy, tlight, theavy, ylight, yheavy = train_test_split(x, t, y, test_size=0.5)
        tlight = tlight[:,1]
        ylight = ylight[:,1]
        theavy = theavy[:,1]
        yheavy = yheavy[:,2]
        # features include treatment and covariates
        t = np.expand_dims(np.hstack([tlight, theavy]), axis=1)
        x = np.hstack([np.vstack([xlight, xheavy]), t])
        y = np.expand_dims(np.hstack([ylight, yheavy]), axis=1)
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size=0.1)

        self.scaler=MinMaxScaler()
        self.scaler.fit(self.xtrain)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.ytrain=self.enc.fit_transform(self.ytrain).toarray()
        self.ytest = self.enc.transform(self.ytest).toarray()
        #print(self.ytest.shape,type(self.ytest))
        self.xtrain = self.scaler.transform(self.xtrain)
        self.xtest = self.scaler.transform(self.xtest)
        self.n_samples,self.n_features=self.xtrain.shape
        self.n_samples_test = self.xtest.shape[0]
        self.shape=[self.n_features]

    def __call__(self, batch_size,batch_index,normalized=True):
        idx_start=batch_index*batch_size
        idx_end=(batch_index+1)*batch_size
        return self.xtrain[idx_start:idx_end,:],self.ytrain[idx_start:idx_end,:]

    def get_test(self):
        return self.xtest, self.ytest

    def get_test_i(self,i):
        return np.reshape(self.xtest[i,:],[1,-1]),np.reshape(self.ytest[i,:],[1,-1])


class NoiseSampler(object):
    def __call__(self, batch_size, z_dim):
        return np.random.normal(size=[batch_size, z_dim])


if __name__ == '__main__':
    #xtrain, ytrain, xtest, ytest = load_mnist()
    x_sampler = IHDPDataSampler()
    xtrain, ytrain = x_sampler(batch_size=32, batch_index=0)
    print('n labels:', len(np.where(np.argmax(x_sampler.ytest,1)==0)[0]),len(np.where(np.argmax(x_sampler.ytest,1)==1)[0]))
