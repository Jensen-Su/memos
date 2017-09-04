import sys
sys.path.append("/home/jcsu/opts/caffe/python/")
import os  
import cPickle  
  
import numpy as np  
from sklearn.model_selection import train_test_split
  
import lmdb  
import caffe  
import cv2

from radial_trans import radial_transform

def unpickle(file):  
    fo = open(file, 'rb')  
    dict = cPickle.load(fo)  
    fo.close()  
    return dict  
  
def shuffle_data(data, labels):  
    data, _, labels, _ = train_test_split(  
        data, labels, test_size=0.0, random_state=42  
    )  
    return data, labels  
  
def load_data(train_file):  
    d = unpickle(train_file)  
    data = d['data']  
    fine_labels = d['fine_labels']  
    length = len(d['fine_labels'])  
  
    data, labels = shuffle_data(  
        data,  
        np.array(fine_labels)  
    )  
    return (  
        data.reshape(length, 3, 32, 32),  
        labels  
    )  
  
if __name__=='__main__':  
        
    x_train, y_train =load_data('cifar-100-python/test') 
    print('Data is fully loaded, now truly processing.')  
    
    grid_size = 4
    shape = x_train.shape  #(num, channels, height, width)
   
    ## stack multiple radial version of an iamge along channel dimension
    x_radial = np.zeros((shape[0] , shape[1]* grid_size * grid_size,
        shape[2], shape[3]), dtype = type(x_train[0, 0, 0, 0]))
    y_radial = np.zeros((shape[0], ), dtype = int)

    grid_r = np.arange(grid_size) * shape[2] / grid_size
    grid_c = np.arange(grid_size) * shape[3] / grid_size

    print('Now performing radial transformation...')
    count = 0
    # for i in range(10): # for each image in the training set
    for i in range(shape[0]): # for each image in the training set
        channel = 0
        for r in grid_r:
            for c in grid_c:
                # stack upon channel dimension
                x_radial[i, channel: channel + 3, ...] = radial_transform(x_train[i], (r, c))
                channel += 3
        
        y_radial[i, ...] = y_train[i]
        count += 1
        if count%10000 == 0:  
            print('already create {} radial pictures'.format(count))  
    
    for i in range(grid_size * grid_size):
        cv2.imwrite("radial_1xkcxhxw" + str(i) + ".jpg", 
                x_radial[np.random.randint(shape[0]), i * 3 : (i+1)*3, ...].transpose(1, 2, 0))

    x_radial, y_radial = shuffle_data(x_radial, y_radial)

    print("Done performing radial transformation.")

    print("Now creating 'radial_1xkcxhxw_lmdb'...")
    env=lmdb.open("test_radial_1xkcxhxw_lmdb",map_size=50000*5000*grid_size*grid_size)  
    txn=env.begin(write=True)  
    count=0  
    for i in range(x_radial.shape[0]):  
        datum=caffe.io.array_to_datum(x_radial[i],y_radial[i])  
        str_id='{:08}'.format(count)  
        txn.put(str_id,datum.SerializeToString())  

        count+=1  
        if count%1000==0:  
            print('already handled with {} pictures'.format(count))  
            txn.commit()  
            txn=env.begin(write=True)  

    txn.commit()  
    env.close()  
    print("Done creating 'radial_1xkcxhxw_lmdb'.")
