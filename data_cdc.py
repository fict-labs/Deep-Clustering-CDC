import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

dataset_name = 'fmnist'   #['fmnist', 'reuters10k', 'stl10', 'cifar10']
use_our_loss = 1
run_mode = 4 

#tuned parameters with grid search: corr_wt, reg_wt, aff_wt: 
weight_params = {'cifar10': [0.000001,  0.001, 0.75], 'fmnist':[0.001, 0.001, 0.75], 'reuters10k': [0.001, 0.001, 0.75],  'stl10': [0.001, 0.001, 0.75]}

#----------- fixed parameters -------------#
dim_z = 10
ae_weight_name = 'ae_weight_' + str(use_our_loss) + '.h5'
cluster_weight_name = 'cluster_weight_' + str(use_our_loss) + '.h5'
kmeans_trials = 20
log_train = 1 

def extract_resnet_features(x):
    #from keras.preprocessing.image import img_to_array, array_to_img
    from keras.utils import img_to_array, array_to_img        
    #from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input    

    im_h = 224
    model = tf.keras.applications.resnet50.ResNet50(include_top=False, pooling='avg', weights='imagenet', input_shape=(im_h, im_h, 3), classifier_activation="none")
    #model.summary()

    print('+ extract_resnet_features...')
    x = np.asarray([img_to_array(array_to_img(im, scale=False).resize((im_h,im_h))) for im in x])
    x = tf.keras.applications.resnet50.preprocess_input(x)  # data - 127. #data/255.#
    features = model.predict(x)
    print('+ Features shape = ', features.shape)
    return features

def load_cifar10(data_path='./data/cifar10'):    
    
    #if features are ready, return them
    import os.path
    if os.path.exists(data_path + '/cifar10_features.npy') and os.path.exists(data_path + '/cifar10_labels.npy'):
        return np.load(data_path + '/cifar10_features.npy'), np.load(data_path + '/cifar10_labels.npy')
    
    from keras.datasets import cifar10
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    x = np.concatenate((train_x, test_x))
    y = np.concatenate((train_y, test_y)).reshape((60000,))

    # extract full features 
    features = np.zeros((60000, 2048))
    for i in range(60):
        idx = range(i*1000, (i+1)*1000)
        print("+ The %dth 1000 samples: " % i)
        features[idx] = extract_resnet_features(x[idx])    
    
    #get 50% random dataset as it is too big for loading in memory
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.5, random_state=11)

    x = X_train
    y = y_train
    print('+ cifar10: x_size = ', x.shape[0])
    print('+ cifar10: y_size = ', y.shape[0])

    # scale to [0,1]
    from sklearn.preprocessing import MinMaxScaler
    x = MinMaxScaler().fit_transform(x)

    #save features
    np.save(data_path + '/cifar10_features.npy', x)
    print('+++  features saved to ' + data_path + '/cifar10_features.npy')

    np.save(data_path + '/cifar10_labels.npy', y)
    print('+++  labels saved to ' + data_path + '/cifar10_labels.npy')

    return x, y

def load_fashion_mnist(data_path='./data/fmnist'):  
    
    from keras.datasets import fashion_mnist  # this requires keras>=2.0.9
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    x = x.reshape((x.shape[0], -1)) #x = (768)
    x = np.divide(x, 255.)

   
    print('+ Fashion fMNIST samples', x.shape)
    return x, y

def load_stl10(data_path='./data/stl', use_resnet = 1):
    use_resnet = 1 #0, 1    

    #NOTE: use_resnet = 0 => never because input dim is too big
    file_name_x = data_path + '/stl_features.npy'
    file_name_y = data_path + '/stl_labels.npy'
    if(use_resnet == 1):
        file_name_x = data_path + '/stl_features_resnet50.npy'
        file_name_y = data_path + '/stl_labels_resnet50.npy'

    import os
    #if features are ready, return them
    if os.path.exists(file_name_x) and os.path.exists(file_name_y):
        return np.load(file_name_x), np.load(file_name_y)
    
    # get labels
    y1 = np.fromfile(data_path + '/train_y.bin', dtype=np.uint8) - 1
    y2 = np.fromfile(data_path + '/test_y.bin', dtype=np.uint8) - 1
    y = np.concatenate((y1, y2))
    
    N = len(y) 
    print("+ load_stl10() => N1 = ", N) #13000
  
    # get data
    x1 = np.fromfile(data_path + '/train_X.bin', dtype=np.uint8)    
    x2 = np.fromfile(data_path + '/test_X.bin', dtype=np.uint8)
    
    # extract features 
    
    if(use_resnet == 1):
        x1 = x1.reshape((int(x1.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
        x2 = x2.reshape((int(x2.size/3/96/96), 3, 96, 96)).transpose((0, 3, 2, 1))
        x = np.concatenate((x1, x2)).astype(float)

        N = len(x) 
        print("+ load_stl10() => N2 = ", N)  #13000
        features = np.zeros((N, 2048))
        for i in range(13):
            idx = range(i*1000, (i+1)*1000)
            print("+ The %dth 1000 samples: " % i)
            features[idx] = extract_resnet_features(x[idx])

        # scale to [0,1]
        from sklearn.preprocessing import MinMaxScaler
        features = MinMaxScaler().fit_transform(features)

        # save features
        np.save(file_name_x, features)
        np.save(file_name_y, y)
        return features, y
    else:
        x = np.concatenate((x1, x2)).astype(float)
        x = np.divide(x, 255.)

        # save features
        np.save(file_name_x, x)
        np.save(file_name_y, y)

        return x, y
    
def make_reuters_data(data_dir):
    np.random.seed(1234)
    from sklearn.feature_extraction.text import CountVectorizer
    from os.path import join
    did_to_cat = {}
    cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
    with open(join(data_dir, 'rcv1-v2.topics.qrels')) as fin:
        for line in fin.readlines():
            line = line.strip().split(' ')
            cat = line[0]
            did = int(line[1])
            if cat in cat_list:
                did_to_cat[did] = did_to_cat.get(did, []) + [cat]
        # did_to_cat = {k: did_to_cat[k] for k in list(did_to_cat.keys()) if len(did_to_cat[k]) > 1}
        for did in list(did_to_cat.keys()):
            if len(did_to_cat[did]) > 1:
                del did_to_cat[did]

    dat_list = ['lyrl2004_tokens_test_pt0.dat',
                'lyrl2004_tokens_test_pt1.dat',
                'lyrl2004_tokens_test_pt2.dat',
                'lyrl2004_tokens_test_pt3.dat',
                'lyrl2004_tokens_train.dat']
    data = []
    target = []
    cat_to_cid = {'CCAT': 0, 'GCAT': 1, 'MCAT': 2, 'ECAT': 3}
    del did
    for dat in dat_list:
        with open(join(data_dir, dat)) as fin:
            for line in fin.readlines():
                if line.startswith('.I'):
                    if 'did' in locals():
                        assert doc != ''
                        if did in did_to_cat:
                            data.append(doc)
                            target.append(cat_to_cid[did_to_cat[did][0]])
                    did = int(line.strip().split(' ')[1])
                    doc = ''
                elif line.startswith('.W'):
                    assert doc == ''
                else:
                    doc += line

    print((len(data), 'and', len(did_to_cat)))
    assert len(data) == len(did_to_cat)

    x = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
    y = np.asarray(target)

    from sklearn.feature_extraction.text import TfidfTransformer
    x = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(x)
    
    from sklearn.model_selection import train_test_split
    x = x.astype(np.float32)    
    print('+ reuters: x_size_full = ', x.shape[0])
    print('+ reuters: y_size_full = ', y.shape[0])

    #extract random 10,000 samples
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=10000, random_state=13)
    x = X_train
    y = y_train
    print('+ reuters: x_size = ', x.shape[0])
    print('+ reuters: y_size = ', y.shape[0])
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
  
    assert x.shape[0] == y.shape[0]
    x = x.reshape((x.shape[0], -1))
    np.save(join(data_dir, 'reutersidf10k.npy'), {'data': x, 'label': y})

def load_reuters_ours(data_path='./data/reuters'):
    import os
    #best_param for this: corr_wt, reg_wt, aff_wt = ['reuters10k_ours': [0.01, 0.1, 1.0]]
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k.npy')):
        print('making reuters idf features')
        make_reuters_data(data_path)
        print(('reutersidf saved to ' + data_path))

    data = np.load(os.path.join(data_path, 'reutersidf10k.npy'), allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('+++++++++ REUTERSIDF10K samples', x.shape))
    return x, y

def load_reuters(data_path='./data/reuters'):
    import os
    if not os.path.exists(os.path.join(data_path, 'reutersidf10k_edesc.npy')):
        print('+++++++++ Download dataset reutersidf10k from EDESC paper first: https://github.com/JinyuCai95/EDESC-pytorch!')
        
    data = np.load(os.path.join(data_path, 'reutersidf10k_edesc.npy'),allow_pickle=True).item()
    # has been shuffled
    x = data['data']
    y = data['label']
    x = x.reshape((x.shape[0], -1)).astype('float64')
    y = y.reshape((y.size,))
    print(('REUTERSIDF10K samples', x.shape))
    return x, y

def load_data(dataset_name):
    if dataset_name == 'fmnist':
        return load_fashion_mnist()
    elif dataset_name == 'stl10':
        return load_stl10()  
    elif dataset_name == 'cifar10':
        return load_cifar10()
    elif dataset_name == 'reuters10k' or dataset_name == 'reuters':
        return load_reuters()   
    else:
        print('Not defined for loading', dataset_name)
        exit(0)