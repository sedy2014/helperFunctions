# from __future__ import print_function
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
def list_dirs(directory):
    """Returns all directories in a given directory
    """
    return [f for f in pathlib.Path(directory).iterdir() if f.is_dir()]
def list_files(directory):
    """Returns all files in a given directory
    """
    return [
        f for f in pathlib.Path(directory).iterdir()
        if f.is_file() and not f.name.startswith(".")
    ]

def create_split(all_files,tr_per,val_per):
    # compute training and validation index
    n_frames = len(all_files)
    file_indices = np.arange(0, n_frames)
    # tr_indx, val_indx = train_test_split(file_indices, train_size=tr_per, test_size=val_per, shuffle=shuffle)
    df = pd.DataFrame(file_indices)
    seed = 4
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(tr_per * m)
    validate_end = int(val_per * m) + train_end
    tr_indx = df.iloc[perm[:train_end]]
    val_indx = df.iloc[perm[train_end:validate_end]]
    test_indx = df.iloc[perm[validate_end:]]
    # convert to numpy array  again
    tr_indx = tr_indx.values.flatten()  # this is preferred to  np.array(tr_indx) . df = [ 1 2] -> [1,2]
    #tr_indx = np.array(tr_indx)   # this does  [ 1 2]-> [[1],[2]]
    val_indx = val_indx.values.flatten()
    test_indx = test_indx.values.flatten()
    return tr_indx,val_indx,test_indx

def move_cpy_files(src_indx,all_files,op_dir,mv):
    for indx in  src_indx :
        src_pth = all_files[indx]._str
        class_nm = os.path.basename(os.path.dirname(src_pth))
        dest_pth = op_dir + "//" + class_nm
        print("src_pth= " + src_pth      + "\n" + "dest_pth= " + dest_pth + "\n")
        if(mv==1):
            shutil.move(src_pth,dest_pth)
        else:
            shutil.copy(src_pth,dest_pth)

#follow: https://pypi.org/project/split-folders/
# bs_pth : shld be a folder with sunfolders, with names specified as list 'class_nms'
#dest_pth  : estinatin folder
# tr_per : training percentage as number less than 1
# val_per : val percentage as number less than 1
# mv: integer 0 (only copy from src to destination folders) or 1: move (to save space)
# CODE TAKES a base path, with files, splits it traiining and validation folders with  subfolders
def tr_va_ts_split(bs_pth,dest_pth,tr_per,val_per,class_nms,mv):
    #bs_pth = 'E://Data//55753//AREDS//jpeg_test'
    train_dir = dest_pth + '//train'
    validation_dir = dest_pth + '//valid'
    test_dir = dest_pth + '//test'
    tst_per = 1 - tr_per - val_per
    # create list of all image data paths
    all_files = []
    for class_dir in list_dirs(bs_pth):
        files_per_class = list_files(class_dir)
        all_files.extend(files_per_class)
    #print(all_files[0])
    #print(all_files)

    # compute training and validation index
    n_frames = len(all_files)
    file_indices = np.arange(0,n_frames)
    #tr_indx, val_indx = train_test_split(file_indices, train_size=tr_per, test_size=val_per, shuffle=shuffle)
    tr_indx,val_indx ,test_indx  = create_split(all_files, tr_per, val_per)

    # read each file at the training index, and write it to output path
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)
    if (tst_per > 0):
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

    # create training, validation and test folders
    #class_nms = ['control','NV','GA']
    for nm in class_nms:
        dest_dir_tr  = train_dir + "//" + nm
        if not os.path.exists(dest_dir_tr):
            os.makedirs(dest_dir_tr)
    for nm in class_nms:
        dest_dir_val  = validation_dir + "//" + nm
        if not os.path.exists(dest_dir_val):
            os.makedirs(dest_dir_val)
    if(tst_per > 0):
        for nm in class_nms:
            dest_dir_tst = test_dir + "//" + nm
            if not os.path.exists(dest_dir_tst):
                os.makedirs(dest_dir_tst)
    else:
        dest_dir_tst = []
    all_files = np.array(all_files)
    move_cpy_files(tr_indx, all_files, train_dir, mv)
    move_cpy_files(val_indx, all_files, validation_dir, mv)
    if (tst_per > 0):
        move_cpy_files(test_indx, all_files, test_dir, mv)
    return all_files,tr_indx,val_indx,test_indx,dest_dir_tr,dest_dir_val,dest_dir_tst



# create subset of train/vald/test data from given data based on percentages
def create_subset(pthin,per): #per = 0.04 # extract this percentage from each class
    from pathlib import Path
    import glob
    for pth, subdir, files in os.walk(pthin):
        if len(files) > 0:
            # get all files with complete path name
            py_files = glob.glob(pth + "//*.png")
            N = int(len(py_files)*per)
            #choose per percentage of files randomly
            ip_files_rand = np.random.choice(py_files,size=N,replace=False)

            # get folder name being train/valdation/test
            tr_val_test = os.path.basename(Path(pth).parents[0])
            # class name, which is subfolder of train/val/test folder
            class_nm = os.path.basename(pth)
            # create op path upto folder name
            op_pth = os.path.join(Path(pthin).parents[0], os.path.basename(pthin) + '_subset', tr_val_test,class_nm)
            if not os.path.exists(op_pth):
                os.makedirs(op_pth)
            print("processing files from " + pth)
            # copy each input file to op path
            for pin in ip_files_rand:
                fnm = os.path.basename(pin)
                pout = os.path.join(op_pth,fnm)
                shutil.copy(pin, pout)

def get_class_stats(pthin): # pthin = 'C:/ML//env//tf//pycharm_areds3//areds//input'
    class_nms = []
    subfolder_names = []
    class_samp = []
    from pathlib import Path
    for pth, subdir, files in os.walk(pthin):
        if len(files) > 0:
            tr_val_test = os.path.basename(Path(pth).parents[0])
            subfolder_names = subfolder_names + [tr_val_test]
            class_samp = class_samp + [len(files)]
            class_nms =  class_nms + [os.path.basename(pth)]
    subfolder_names = np.unique(subfolder_names)
    N_subfolders = len(subfolder_names)
    # get class samples as 2D array, each row is a samples from a folder tr/val/test
    class_samples = np.array(class_samp)
    class_samples = np.reshape(class_samples, (N_subfolders, int(len(class_samples)/N_subfolders)))
    # normalized with respect to sum of class samples, for each folder(row)
    class_samples_norm = class_samples/class_samples.sum(axis=1, keepdims=True) * 100
    class_names  = np.array(class_nms)
    class_names = np.reshape(class_names, (N_subfolders, int(len(class_names)/N_subfolders)))
    return subfolder_names, class_names, class_samples ,class_samples_norm
# get numpy array of each class count

#***********************  print batch data to files *************************
# next(train_gen) gives batch, where batch[0] is (#batchsize,W,H,C) and batch[1] is (2,1) or yop or each class
# this needs to be exceuted to get each batch stats,like current index
# train_gen.index_array gives indices of all samples after shuffling
# train_gen.filepaths gives filepaths of samples without shuffle
def save_batch_info(gen,pth):
    batches_per_epoch = int(np.ceil(gen.samples // gen.batch_size )) + 1
    #with open ('./glucoma/saved_plots/glucoma_v2_batch_tr.txt',"w")as fp:
    with open(pth, "w")as fp:
        for i in range(batches_per_epoch):
            # get each batch from training generator
            batch = next(gen)
            current_index = ((gen.batch_index-1) * gen.batch_size)
            if current_index < 0:
                if gen.samples % gen.batch_size > 0:
                    current_index = max(0,gen.samples - gen.samples % gen.batch_size)
                else:
                    current_index = max(0,gen.samples - gen.batch_size)
            index_array = gen.index_array[current_index:current_index + gen.batch_size]
            cu_btch = "batch" + str(i)
            fp.write(cu_btch + "\n")
            for idx in index_array:
                cu_pth = gen.filepaths[idx]
                fp.write(cu_pth + "\n")

def get_batch_stats(gen,pth_saved_plots):
    from pathlib import Path
    batch_x_all = []
    # for batch_x,batch_y in gen:
    #     batch_x_all = batch_x_all + batch_x
    # somehow, i need to add 1 to cover all samples
    batches_per_epoch = int(np.ceil(gen.samples // gen.batch_size )) + 1
    # 2-D array of size number of batches * class freq in each batch
    class_freq_usd_batch_all =  np.zeros((batches_per_epoch,len(gen.class_indices)))
    sum_batch_all = 0
    #class_nms_batch_all = np.chararray((batches_per_epoch, len(gen.class_indices)))
    for i in range(batches_per_epoch):
        # get each batch from training generator
        batch = next(gen)
        current_index = ((gen.batch_index-1) * gen.batch_size)
        if current_index < 0:
            if gen.samples % gen.batch_size > 0:
                current_index = max(0,gen.samples - gen.samples % gen.batch_size)
            else:
                current_index = max(0,gen.samples - gen.batch_size)
        index_array = gen.index_array[current_index:current_index + gen.batch_size]
        cu_btch = "batch" + str(i)
        class_arr_per_batch = []
        for idx in index_array:
            cu_pth = gen.filepaths[idx]
            class_arr_per_batch = class_arr_per_batch + [os.path.basename(Path(cu_pth).parents[0])]
            # get class names array for current batch
        class_arr_per_batch = np.array(class_arr_per_batch)
        # get distinct class names, their frequency in current batch
        class_nms_usd_batch,b, class_freq_usd_batch= np.unique(class_arr_per_batch, return_index=True, return_counts=True)
        class_nms = []
        # get all classes in train/valid folder
        for k1, v1 in gen.class_indices.items():
            class_nms = class_nms + [k1]
        class_nms_unusd_batch = np.setdiff1d(class_nms, class_nms_usd_batch.tolist())
        # create class names array again, with unused classes appended at the end
        class_nms_batch =  np.hstack((class_nms_usd_batch, class_nms_unusd_batch))
        # add zero frequency for classes in batch whose samples were not picked up
        class_freq_usd_batch = np.hstack((class_freq_usd_batch, np.zeros(len(class_nms_unusd_batch))))
        sum_batch_all = sum_batch_all + class_freq_usd_batch.sum()
        class_freq_usd_batch = class_freq_usd_batch/class_freq_usd_batch.sum()
        # create class freq as per original class names in order
        ind_arr = []
        cnt = 0
        class_freq_usd_batch_aligned =  np.zeros((len(class_freq_usd_batch),))
        for k,v in gen.class_indices.items():
            # find key(class) name in new array
            ind =  np.where(class_nms_batch == k)
            class_freq_usd_batch_aligned[cnt] = class_freq_usd_batch[ind]
            cnt = cnt + 1
        class_freq_usd_batch_all[i, ] = class_freq_usd_batch_aligned
    # check that total number of samples in all classes computed above match the one from training generator
    if (sum_batch_all != len(gen.filepaths)):
        print('some batch samples were missed while plotting')
    # all stats are now computed
    # for each class, compute histogram, which is column of class_freq_usd_batch_all array
    j = 0
    for k,v in gen.class_indices.items():
        # get each column of data
        class_occur = class_freq_usd_batch_all[ :,j]
        #class_hist = np.histogram(class_occur, bins= np.arange(0.1,1.1,0.1))
        plt.hist(class_occur,density=True)
        plt.title('all_batch_histogram_class_' + k)
        plt.savefig(os.path.dirname(pth_saved_plots) + '\\' + 'all_batch_histogram_class_' + k + '.png')
        j = j + 1



# smoothen training accuracy/loss data as per 1st order MA
def smooth_curve(data,alpha):
    smooth_d = []
    for point in data:
        if smooth_d:
            prev = smooth_d[-1]
            smooth_d.append((prev * alpha) + point * (1 - alpha))
        else:
            smooth_d.append(point)
    return smooth_d

# sve model hyperparams
def save_model_hyperparam(model,pth_saved_model,pth_batch_tr,pth_batch_vld,load_mod,save_model_weights,save_best_weights,train_gen,vd_gen,batch_size,epochs,lr_st,loss_func,optimizer,callbacks_list):
    d1 = model.get_config()
    with open(pth_saved_model + '//param.txt', 'w') as f:
        # for i in d1['layers']:
        #     print(i, file=f)
        #     f.write('\n')
        f.write('pth_batch_tr = ' + str(pth_batch_tr) + '\n')
        f.write('pth_batch_vld = ' + str(pth_batch_vld) + '\n')
        f.write('load_mod = ' + str(load_mod) + '\n')
        f.write('save_model_weights = ' + str(save_model_weights) + '\n')
        f.write('save_best_weights = ' + str(save_best_weights) + '\n')
        f.write('train_gen_resize = ' + str(train_gen.target_size) + '\n')
        f.write('valid_gen_resize = ' + str(vd_gen.target_size) + ' \n')
        f.write('batchsize = ' + str(batch_size) + '\n')
        f.write('epochs = ' + str(epochs) + "\n")
        f.write('lr_st = ' + str(lr_st) + "\n")
        f.write('loss_func = ' + str(loss_func) + "\n")
        f.write('optimizer = ' + str(optimizer) + "\n")
        f.write('callbacks = ' + str(callbacks_list) + "\n")
        f.close()

# save accuracy and loss training plots with and without smoothing
def save_plots(mdlfit,pth_saved_plots,fnm,epochs):
    import matplotlib.pyplot as plt
    trn_acc = mdlfit.history['acc']
    val_acc = mdlfit.history['val_acc']
    lr_ep = mdlfit.history['lr']
    epochs = range(len(trn_acc))
    plt.plot(epochs,trn_acc,'b-',label = 'Train accuracy')
    plt.plot(epochs,val_acc,'g-',label = 'Valid accuracy')
    plt.title('Accuracy')
    plt.legend(loc='center right')

    plt.savefig(pth_saved_plots + '/' + fnm  +  '_acc.png')
    #plt.show()
    plt.close()
    plt.plot(epochs,smooth_curve(trn_acc,0.8),'b-',label = 'Smoothed Train accuracy')
    plt.plot(epochs,smooth_curve(val_acc,0.8),'g-',label = 'Smoothed Valid accuracy')
    plt.title('Smoothed Accuracy')
    plt.legend(loc='center right')
    plt.savefig(pth_saved_plots + '/' + fnm  +  '_smooth_acc.png')
    #plt.show()
    plt.close()
    #plot training and validtion loss
    trn_loss = mdlfit.history['loss']
    val_loss = mdlfit.history['val_loss']
    epochs = range(len(trn_loss))
    plt.plot(epochs,trn_loss,'b-',label = 'Train Loss')
    plt.plot(epochs,val_loss,'g-',label = 'Valid Loss')
    plt.title('Loss')
    plt.legend(loc='center right')
    plt.savefig(pth_saved_plots + '/' + fnm  +  '_loss.png')
    #plt.show()
    plt.close()
    plt.plot(epochs,smooth_curve(trn_loss,0.8),'b-',label = 'Smoothed Train Loss')
    plt.plot(epochs,smooth_curve(val_loss,0.8),'g-',label = 'Smoothed Valid Loss')
    plt.title('Smoothed Loss')
    plt.legend(loc='center right')
    plt.savefig(pth_saved_plots + '/' + fnm  +  '_smooth_loss.png')
    #plt.show()
    plt.close()

    plt.plot(epochs, lr_ep, 'b-', label='Learning Rate')
    plt.title('Learning_Rt')
    plt.legend(loc='center right')
    plt.savefig(pth_saved_plots + '/' + fnm + '_lr.png')

    np.savetxt(pth_saved_plots + '/' + fnm + '_all.txt', np.vstack(((trn_acc, val_acc), (trn_loss, val_loss))).T)
    np.savetxt(pth_saved_plots + '/' + fnm + '_lr.txt', lr_ep)