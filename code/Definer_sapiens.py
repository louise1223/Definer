import random
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, Input,MaxPooling1D,Flatten,LeakyReLU,Activation,concatenate,Reshape,GRU,LSTM,Add,attention,Multiply
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import keras
import os
np.random.seed(seed=21) #序列长度

def analyze(temp, OutputDir):
    trainning_result, validation_result, testing_result = temp;
    file = open(OutputDir + '/result对比实验onehot.txt', 'w')
    index = 0
    for x in [trainning_result, validation_result, testing_result]:
        title = ''
        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'
        index += 1;
        file.write(title +  'results\n')
        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:
            total = []
            for val in x:
                total.append(val[j])
            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n')

        file.write('\n\n______________________________\n')
    file.close();
    index = 0
    for x in [trainning_result, validation_result, testing_result]:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            i += 1
        print;

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;


def scheduler(epochs, lr):
  if epochs < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

    
def check_sequence(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    
    return out

def calculate(sequence):

    X = []
    dictNum = {'A' : 0, 'U' : 0, 'C' : 0, 'G' : 0};

    for i in range(len(sequence)):

        if sequence[i] in dictNum.keys():
            dictNum[sequence[i]] += 1;
            X.append(dictNum[sequence[i]] / float(i + 1));

    return np.array(X)


def Encoded_onehot(path, T):
    data = pd.read_csv(path);

    X = [];
    data = data.values
    for line in range(0, len(data), 2):
        seq_data = []
        seq = data[line][0]
        for i in range(len(seq)):
            if seq[i] == 'A':
                seq_data.append([1, 0, 0, 0])
            if seq[i] == 'U':
                seq_data.append([0, 1, 0, 0])
            if seq[i] == 'C':
                seq_data.append([0, 0, 1, 0])
            if seq[i] == 'G':
                seq_data.append([0, 0, 0, 1])
            if seq[i] == 'N':
                seq_data.append([0, 0, 0, 0])

        X.append(np.array(seq_data));
    if T == "P":
        y = [1] * len(X)
    if T == "N":
        y = [0] * len(X)
    X = np.array(X);
    y = np.array(y, dtype=np.int32);
    return X, y;


def Encoded_NCP(path, T):
    dataset = pd.read_csv(path, header=None)
    X = []
    for i in dataset.iloc[:, 0]:
        if ">" in i:
            continue
        seq_data = []
        for j in range(len(i)):
            if i[j] == "A":
                seq_data.append([1, 1, 1])
            elif i[j] == "C":
                seq_data.append([0, 1, 0])
            elif i[j] == "G":
                seq_data.append([1, 0, 0])
            elif i[j] == "T":
                seq_data.append([0, 0, 1])
            elif i[j] == "U":
                seq_data.append([0, 0, 1])
            else:
                seq_data.append([0, 0, 0])
        X.append(seq_data)
    if T == "P":
        y = [1] * len(X)
    if T == "N":
        y = [0] * len(X)
    X = np.array(X);
    y = np.array(y, dtype=np.int32);
    return X, y;


def Datasets(PositiveData, NegativeData):
    Positive_X_hot, Positive_y_hot = Encoded_onehot(PositiveData, "P");
    Negitive_X_hot, Negitive_y_hot = Encoded_onehot(NegativeData, "N");
    Positive_X_ncp, Positive_y_ncp = Encoded_NCP(PositiveData, "P");
    Negitive_X_ncp, Negitive_y_ncp = Encoded_NCP(NegativeData, "N");
    #
    Positive_X = np.concatenate([Positive_X_hot, Positive_X_ncp], axis=-1)
    Negitive_X = np.concatenate([Negitive_X_hot, Negitive_X_ncp], axis=-1)
    Positive_y = Positive_y_hot
    Negitive_y = Negitive_y_ncp

    # Positive_X, Positive_y = Encoded_onehot(PositiveData, "P");
    # Negitive_X, Negitive_y = Encoded_onehot(NegativeData, "N");

    # Positive_X, Positive_y = Encoded_NCP(PositiveData, "P");
    # Negitive_X, Negitive_y = Encoded_NCP(NegativeData, "N");

    return Positive_X, Positive_y, Negitive_X, Negitive_y


def splitData(X):
    hot = X[:, :, :4]
    ncp = X[:, :, 4:]

    return hot, ncp


def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;


def Att(inputs):
    V = inputs
    QK = Dense(64)(inputs)
    QK = Activation(activation='relu')(QK)
    MV = Multiply()([V, QK])
    return MV
def model_Definer():
    input_shape = (21, 4)
    inputs = Input(shape=input_shape)

    conv0 = Conv1D(filters=32, kernel_size=5, strides=1)(inputs)
    normLayer0 = BatchNormalization()(conv0);
    act0 = Activation(activation='relu')(normLayer0)

    gru = GRU(64, return_sequences=True)(act0)

    conv2 = Conv1D(filters=64, kernel_size=5, strides=1)(act0)
    normLayer2 = BatchNormalization()(conv2);
    pool2 = MaxPooling1D(pool_size=2)(normLayer2)
    dropoutLayer1 = Dropout(0.35)(pool2)
    act2 = Activation(activation='relu')(dropoutLayer1)
    res = concatenate([act2, gru], axis=1)

    x = Flatten()(res)

    conv3 = Conv1D(filters=64, kernel_size=5, strides=1)(act0)
    normLayer4 = BatchNormalization()(conv3);
    pool4 = MaxPooling1D(pool_size=2)(normLayer4)
    dropoutLayer2 = Dropout(0.35)(pool4)
    act4 = Activation(activation='relu')(dropoutLayer2)
    res_2 = concatenate([act4, gru], axis=1)

    res_2 = Att(res_2)
    Definer = Flatten()(res_2)
    comb = concatenate([x, Definer], axis=1)

    Definer1 = keras.layers.Lambda(lambda comb: comb[:, 0:384], output_shape=(384,))(comb)
    Definer2 = keras.layers.Lambda(lambda comb: comb[:, 384:], output_shape=(384,))(comb)

    Definer1 = Dense(8, activation='relu')(Definer1)
    Definer2 = concatenate([Definer2, Definer1])
    Definer2 = Dense(8, activation='relu')(Definer2)
    Definer3 = concatenate([Definer1, Definer2])
    Definer3 = Dense(8, activation='relu')(Definer3)
    Definer4 = concatenate([Definer2, Definer3])
    Definer4 = Dense(8, activation='relu')(Definer4)
    Definer5 = concatenate([Definer1, Definer4])
    Definer5 = Dense(8, activation='relu')(Definer3)
    Definer6 = concatenate([Definer2, Definer5])
    Definer6 = Dense(8, activation='relu')(Definer6)
    Definer = concatenate([Definer1, Definer2], axis=1)
    Definer = concatenate([Definer, Definer3], axis=1)
    Definer = concatenate([Definer, Definer4], axis=1)
    Definer = concatenate([Definer, Definer5], axis=1)
    Definer = concatenate([Definer, Definer6], axis=1)

    output = Dense(1, activation='sigmoid')(Definer)

    model = Model(inputs=inputs, outputs=output)
    opt = SGD(learning_rate=0.001, momentum=0.95)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[binary_accuracy]);


def calculateScore(X, y, model, folds):
    
    pred_y = model.predict(X)

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    # 保存结果标签
    pre_result = []

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
        pre_result.append(tempLabel[i])
        # 将结果保存到文件，自己make
    with open("../test/pre_result.txt", "w") as f:
        for i in pre_result:
            if i == 1:
                f.write("Yes\n")
            else:
                f.write("No\n")

    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN+FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    # plt.show()
    
    lossValue = losses.binary_crossentropy(y_true, y_pred)#

def funciton(PositiveCSV, NegativeCSV, OutputDir, folds, valib_P=None, valib_N=None):

    Positive_X, Positive_y, Negitive_X, Negitive_y = Datasets(PositiveCSV, NegativeCSV)
    
    random.shuffle(Positive_X);
    random.shuffle(Negitive_X);

    Positive_X_Slices = check_sequence(Positive_X, folds);
    Positive_y_Slices = check_sequence(Positive_y, folds);

    Negative_X_Slices = check_sequence(Negitive_X, folds);
    Negative_y_Slices = check_sequence(Negitive_y, folds);

    trainning_result = []
    validation_result = []
    testing_result = []

    # 导入数据处理
    valib_P_X, valib_P_y, valib_N_X, valib_N_y = Datasets(valib_P, valib_N)

    random.shuffle(valib_P_X);
    random.shuffle(valib_N_X);

    Valib_P_X_Slices = check_sequence(valib_P_X, folds);
    Valib_P_y_Slices = check_sequence(valib_P_y, folds);

    Valib_N_X_Slices = check_sequence(valib_N_X, folds);
    Valib_N_y_Slices = check_sequence(valib_N_y, folds);

    for test_index in range(folds):

        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Valib_P_X_Slices[0], Valib_N_X_Slices[0]))
        valid_y = np.concatenate((Valib_P_y_Slices[0], Valib_N_y_Slices[0]))



        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))
        # print(train_X.shape)
        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))
                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))
        # print(np.shape(tempX),np.shape(train_X))
        test_X, test_y = shuffleData(test_X,test_y);
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y);

        model = model_Definer();

        result_folder = OutputDir
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        model_results_folder=result_folder
        
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30, restore_best_weights=True)
        model_check = ModelCheckpoint(filepath = OutputDir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True, save_weights_only=False)
        
        reduct_L_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        cbacks = [model_check, early_stopping,reduct_L_rate]


        model = load_model('../Definer_model/sapiens_result/model10.h5')

        trainning_result.append(calculateScore(train_X, train_y, model, folds));
        validation_result.append(calculateScore(valid_X, valid_y, model, folds));
        testing_result.append(calculateScore(test_X, test_y, model, folds));


def Valib_data(OutputDir,valib_P=None, valib_N=None):

    validation_result = []
    # 导入数据处理
    valib_P_X, valib_P_y, valib_N_X, valib_N_y = Datasets(valib_P, valib_N)
    random.shuffle(valib_P_X);
    random.shuffle(valib_N_X);
    Valib_P_X_Slices = check_sequence(valib_P_X,1);
    Valib_P_y_Slices = check_sequence(valib_P_y,1);
    Valib_N_X_Slices = check_sequence(valib_N_X,1);
    Valib_N_y_Slices = check_sequence(valib_N_y,1);
    valid_X = np.concatenate((Valib_P_X_Slices[0], Valib_N_X_Slices[0]))
    valid_y = np.concatenate((Valib_P_y_Slices[0], Valib_N_y_Slices[0]))


    valid_X, valid_y = shuffleData(valid_X, valid_y)
    result_folder = OutputDir
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model = load_model('../Definer_model/sapiens_result/model10.h5')
    validation_result.append(calculateScore(valid_X, valid_y, model, 10));
    # print("结果显示：", validation_result)
def out_(valib_P=None, valib_N=None):

    NegativeCSV = r"..\data\sapiens_negitive_495.txt"
    PositiveCSV = r"..\data\sapiens_positive_495.txt"

    OutputDir = r"..\Definer_model\sapiens_result"

    Valib_data(OutputDir, valib_P, valib_N)
    print("本次程序已经结束，欢迎使用。")

if __name__=="__main__":
    out_()








