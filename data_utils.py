import numpy as np
import scipy.stats,scipy.special
import os, sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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
        t, y, y_ctf = data[:, 0][:, np.newaxis], data[:, 1][:, np.newaxis], data[:, 2][:, np.newaxis]
        #x = np.hstack([t, data[:, 5:]]) # fields 3 and 4 are mus and not used.
        x = data[:, 5:]
        self.xtrain, self.xtest, self.ttrain, self.ttest, self.ytrain, self.ytest, self.yctftrain, self.yctftest = train_test_split(x, t, y, y_ctf, test_size=0.1)

        self.n_samples,self.n_features=self.xtrain.shape
        self.n_samples_test = self.xtest.shape[0]
        self.shape=[self.n_features]

    def __call__(self, batch_size,batch_index,normalized=True,ctf=False):
        idx_start=batch_index*batch_size
        idx_end=(batch_index+1)*batch_size
        if ctf: 
            return self.xtrain[idx_start:idx_end,:],self.ytrain[idx_start:idx_end,:],self.ttrain[idx_start:idx_end,:],self.yctftrain[idx_start:idx_end,:]
        else:
            return self.xtrain[idx_start:idx_end,:],self.ytrain[idx_start:idx_end,:]

    def get_test(self):
        return self.xtest, self.ytest

    def get_test_i(self,i):
        return np.reshape(self.xtest[i,:],[1,-1]),np.reshape(self.ytest[i,:],[1,-1])

def preprocess(dataframe,data_types_file):
    s = open(data_types_file, 'r').read()
    types_dict = eval(s)

    column_names = sorted(dataframe.columns)
    dataframe = dataframe[column_names]
    data = np.asarray(dataframe)
    #Construct the data matrices
    data_complete = []
    column_names_new = []
    for i,column in enumerate(column_names):
        if types_dict[column] == 'cat' or types_dict[column]=='ord' or types_dict[column] == 'cyc':
            #Get categories
            cat_data = [int(x) for x in data[:,i] if ~np.isnan(x)]
            categories, indexes = np.unique(cat_data,return_inverse=True)
            #Transform categories to a vector of 0:n_categories
            new_categories = np.arange(int(len(categories)))
            cat_data = new_categories[indexes]
            #Create one hot encoding for the categories
            aux = np.zeros([np.shape(data)[0],len(new_categories)])
            aux[np.where(~np.isnan(data[:,i]))[0],cat_data] = 1
            data_complete.append(aux)
            column_names_new.extend([column_names[i]+'_'+ str(kk) for kk in range(len(new_categories))])
        else:
            imp = SimpleImputer(strategy='mean')
            data_vec = imp.fit_transform(np.reshape(data[:,i],[np.shape(data)[0],1]))
            data_vec = np.reshape(data_vec,[np.shape(data)[0],1])
            data_complete.append(data_vec)
            column_names_new.append(column_names[i])

    data_new = np.concatenate(data_complete,1)
    column_names_new = column_names_new
    data_new = pd.DataFrame(data_new,columns=column_names_new)
    return data_new


class TwinsDataSampler(object):
    def __init__(self, simulate_confounding=1.0):
        self.n_labels = 2
        
        # select the 46 columns from CEVAE paper
        #fields = ['adequacy', 'alcohol', 'anemia', 'birattnd', 'brstate', 'brstate_reg', 'cardiac', 'chyper', 'cigar6', 'crace', 'csex', 'dfageq', 'diabetes', 'dlivord_min', 'dmar', 'drink5', 'dtotord_min', 'eclamp', 'feduc6', 'frace', 'gestat10', 'hemo', 'herpes', 'hydra', 'incervix', 'lung', 'mager8', 'meduc6', 'mplbir', 'mplbir_reg', 'mpre5', 'mrace', 'nprevistq', 'orfath', 'ormoth', 'othermr', 'phyper', 'pldel', 'pre4000', 'preterm', 'renal', 'rh', 'stoccfipb', 'stoccfipb_reg', 'tobacco', 'uterine']

        s = open('../CEVAE-master/datasets/TWINS/covar_type.txt', 'r').read()
        fields = sorted(list(eval(s).keys()))
        fields.remove('bord')
        fields.remove('infant_id')

        x_raw = pd.read_csv('../CEVAE-master/datasets/TWINS/twin_pairs_X_3years_samesex.csv',usecols=fields)
        t_raw = pd.read_csv('../CEVAE-master/datasets/TWINS/twin_pairs_T_3years_samesex.csv',usecols=['dbirwt_0','dbirwt_1'])
        y_raw = pd.read_csv('../CEVAE-master/datasets/TWINS/twin_pairs_Y_3years_samesex.csv',usecols=['mort_0','mort_1'])
        x_raw = x_raw.where(pd.notnull(x_raw), np.nan)
        x_raw = x_raw[fields]

        # select rows where both twins <2kg at birth
        low_weight = (t_raw["dbirwt_0"]<2000) & (t_raw["dbirwt_1"]<2000)
        self.x = np.asarray(x_raw[low_weight])
        self.t = np.asarray(t_raw[low_weight])
        self.y = np.asarray(y_raw[low_weight])

        idx_valid = np.union1d(np.where(~np.isnan(self.t).any(axis=1))[0],np.where(~np.isnan(self.y).any(axis=1))[0])
        #should be all the rows
        self.x = self.x[idx_valid]
        self.t = self.t[idx_valid]
        self.y = self.y[idx_valid]

        data_filtered = pd.DataFrame(self.x,columns=fields)
        data_new= preprocess(data_filtered,'../CEVAE-master/datasets/TWINS/covar_type.txt')

        if not simulate_confounding:
            #simulates rct - if 1 choose heavier twin, choose lighter otherwise
            self.ttrain = np.random.choice([0,1],size=[self.x.shape[0],1],replace=True)
            self.wtrain = np.asarray([self.t[idx,p] for idx,p in zip(range(self.t.shape[0]),self.ttrain)])
            self.ytrain = np.asarray([self.y[idx,p] for idx,p in zip(range(self.y.shape[0]),self.ttrain)])
            self.xtrain = np.asarray(data_new)
            self.wtest = np.asarray([self.t[idx,p] for idx,p in zip(range(self.t.shape[0]),1-self.ttrain)])
            self.ytest= np.asarray([self.y[idx,p] for idx,p in zip(range(self.y.shape[0]),1-self.ttrain)])
            self.xtest = np.asarray(data_new)
            self.ttest = 1-self.ttrain
        else:
            idx = [i for i in range(len(x_raw.columns)) if  x_raw.columns[i]!='gestat10']
            idx_g = np.setdiff1d(range(len(x_raw.columns)),idx)
            w_0 = np.random.normal(0,0.1,size=len(idx))
            w_h = np.random.normal(5,0.1,size=1)
            #scaler = StandardScaler()
            #tmp_idx = scaler.fit_transform(foo[:,idx])
            #pvec = scipy.special.expit(((w_h*self.x[:,idx_g])/(10-0.1)).flatten()+ np.matmul(tmp_idx,w_0))
            pvec = scipy.special.expit((w_h*self.x[:,idx_g]/(10-0.1)).flatten()).flatten()
            idx_valid = np.where([not np.isnan(f) for f in pvec])[0]
            self.ttrain = np.asarray([np.random.binomial(1,pp,1) for pp in pvec[idx_valid]])
            self.xtrain = np.asarray(data_new)[idx_valid,:]
            self.ytrain = np.asarray([self.y[idx,p] for idx,p in zip(idx_valid,self.ttrain)])
            self.wtrain = np.asarray([self.t[idx,p] for idx,p in zip(idx_valid,self.ttrain)]) 
            self.ttest = 1-self.ttrain
            self.xtest = np.asarray(data_new)[idx_valid,:]
            self.ytest = np.asarray([self.y[idx,p] for idx,p in zip(idx_valid,self.ttest)])
            self.wtest = np.asarray([self.t[idx,p] for idx,p in zip(idx_valid,self.ttest)])

        self.scaler=StandardScaler()
        self.scaler.fit(self.xtrain)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.ytrain=self.enc.fit_transform(self.ytrain).toarray()
        self.ytest = self.enc.transform(self.ytest).toarray()
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
    xs = TwinsDataSampler()
    #print(xs.xtrain.shape,xs.ytrain.shape,xs.ttrain.shape,xs.xtest.shape)
    #xtrain, ytrain = x_sampler(batch_size=32, batch_index=0)
    #print('n labels:', len(np.where(np.argmax(x_sampler.ytest,1)==0)[0]),len(np.where(np.argmax(x_sampler.ytest,1)==1)[0]))

