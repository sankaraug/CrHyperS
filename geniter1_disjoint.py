import torch
import numpy as np
import torch.utils.data as Data

def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length,  dimension, gt):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    gt_cubic_data = np.zeros(data_size)
    data_assign = data_indices
    for i in range(len(data_assign)):
        small_cubic_data[i] = whole_data[data_indices[i],:,:,:]
        gt_cubic_data[i] = gt[data_indices[i]] 
    return small_cubic_data, gt_cubic_data


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices,
                  whole_data, PATCH_LENGTH, INPUT_DIMENSION, batch_size, gt):
    #gt_all = gt[total_indices] - 1
    #y_train = gt[train_indices] - 1
    #y_test = gt[test_indices] - 1
    
    gt = gt -1

    all_data, gt_all =  select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH,  INPUT_DIMENSION, gt)
    

    train_data, y_train = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH,  INPUT_DIMENSION, gt)
    print(train_data.shape, y_train.shape)
    
    test_data, y_test =  select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH,  INPUT_DIMENSION, gt)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_test = x_test_all
 
    
    """
    all_data = whole_data
    gt_all  = gt
    from sklearn.model_selection import train_test_split
    x_train, Xtest, y_train, ytest = train_test_split(whole_data, gt, test_size= 0.7, random_state=7, stratify=gt)
    x_train.shape, Xtest.shape, y_train.shape, ytest.shape
    x_val, x_test, y_val, y_test = train_test_split(Xtest, ytest, test_size= 0.5, random_state=7, stratify=ytest)
    x_val.shape, x_test.shape, y_val.shape, y_test.shape """
    
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    
    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=0, 
    )
    
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False, 
        num_workers=0, 
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False, 
        num_workers=0, 
    )
    return train_iter, test_iter, all_iter #, y_test


def generate_iter_test(TRAIN_SIZE, train_indices, whole_data, PATCH_LENGTH, INPUT_DIMENSION, batch_size, gt):
    gt = gt -1
    #train_data = whole_data
    #y_train =  gt
    train_data, y_train = select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH,  INPUT_DIMENSION, gt)
    print("Test data shape:")
    print(train_data.shape, y_train.shape)
    
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  
        num_workers=0, 
    )

    return train_iter
