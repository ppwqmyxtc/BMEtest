import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        # nn.init.kaiming_uniform_(m.weight)
        nn.init.xavier_uniform_(m.weight)


# LSTM模型定义
class LSTMClass(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTMClass, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,  num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # self.softmax_o = nn.LogSoftmax(dim=1)
        # self.relu = nn.ReLU()

    def forward(self, x, y, h0, c0):
        out, _ = self.lstm(x, (h0, c0))
        # 通过全连接层进行预测
        pred = self.sigmoid(self.output_layer(out[:, -1, :]))
        y = torch.FloatTensor(y)
        return pred, y


# LSTM程序定义
def LSTMmodel(train_x, train_y, test_x, test_y):
    input_size = 33  # 特征个数
    timestep = 12  # 时间步
    hidden_size = 128  # 隐藏层单元个数
    num_layers = 2  # 网络层数
    output_size = 1  # 输出维度
    learning_rate = 0.0001  # 学习率
    l2 = 0.001  # L2正则化
    Epoch = 100  # 迭代次数
    batch_size = 32  # 批次量大小
    sample_num = 10
    n_batches = int(np.floor(float(len(train_x)) / float(batch_size)))
    # print('---n_batches:', n_batches)

    model = LSTMClass(input_size, hidden_size, num_layers, output_size, batch_size)
    model.train()
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)  #
    loss_all = []
    train_cost_list = []
    # print('-------------train:')
    # 初始化隐状态和单元状态
    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    c0 = torch.zeros(num_layers, batch_size, hidden_size)
    for epoch in range(Epoch):
        samples = random.sample(range(n_batches), sample_num)
        for index in samples:
            batch_matrix = train_x[batch_size * index: batch_size * (index + 1)].reshape(batch_size, input_size, -1)
            batch_matrix = torch.FloatTensor(np.transpose(batch_matrix, (0, 2, 1)))
            batch_labels = train_y[batch_size * index: batch_size * (index + 1)]
            pred, labels = model(batch_matrix, batch_labels, h0, c0)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(pred.squeeze(), labels)
            loss.backward()
            optimizer.step()
            loss_all.append(loss.cpu().data.numpy())
        train_cost = np.mean(loss_all)
        train_cost_list.append(train_cost)
        # if (epoch % 10) == 0:
        #     print('----epoch:', epoch)
    # plt.figure()
    # plt.plot(train_cost_list)
    # plt.title('Loss curve')
    # plt.xlabel('Interations')
    # plt.ylabel('Loss Curve')
    # plt.show()
    # Save model.
    # torch.save(model, 'data_save/lstm_model.m')
    # print('-------------test:')
    model.eval()
    test_x = torch.FloatTensor(np.transpose(test_x.reshape(-1, input_size, timestep), (0, 2, 1)))
    print('test shape:', test_x.shape)
    h0_test = torch.zeros(num_layers, test_x.shape[0], hidden_size)
    c0_test = torch.zeros(num_layers, test_x.shape[0], hidden_size)
    pred_test, labels_test = model(test_x, test_y, h0_test, c0_test)
    pred_test = np.array(pred_test.detach().numpy())
    fpr, tpr, thresholds = metrics.roc_curve(np.array(labels_test.detach().numpy()), pred_test, pos_label=1)
    # plt.plot(fpr, tpr)
    # plt.show()
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    print('LSTM AUROC:', roc_auc)
    gmean = np.sqrt(tpr * (1 - fpr))
    # 查找最佳阈值
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

    pred_test_label = (pred_test >= thresholdOpt)  # 将得分转化为对应的0&1值
    pred_test_label = [int(item) for item in pred_test_label]
    pred_test_label = np.array(pred_test_label).reshape([-1, 1])
    writer1 = pd.ExcelWriter('results_save/results_lstm_pred.xlsx')
    pd.DataFrame(np.array(pred_test).reshape([-1, 1])).to_excel(writer1, sheet_name='pred_label_score')
    pd.DataFrame(np.array(pred_test_label).reshape([-1, 1])).to_excel(writer1, sheet_name='pred_label')
    writer1._save()
    auc_roc, sen, spe, acc, auc_prc, recall, precision, f1 = calculate_performance(np.array(labels_test.detach().numpy()), pred_test.squeeze(), pred_test_label.squeeze())
    return auc_roc, sen, spe, acc, auc_prc, recall, precision, f1


def LRmodel(train_x, train_y, test_x, test_y):
    num_pos = sum(i == 1 for i in train_y)
    num_neg = sum(i == 0 for i in train_y)
    # rate = 1.0 * num_pos / num_neg  ## 后续处理类不平衡问题时所使用的
    # if flag_resample:
    #     # resampled_train_x, resampled_train_y = resampling(inter_train_x, inter_train_y, sample_method='undersampling')
    #     resampled_train_x, resampled_train_y = resampling(inter_train_x, inter_train_y, sample_method='oversampling')
    #     trainmodel_x =\
    #         resampled_train_x
    #     trainmodel_y = resampled_train_y
    #     inter_train_x =  trainmodel_x
    #     inter_train_y =  trainmodel_y
    log = LogisticRegression(solver='saga', max_iter=1000, C=0.01)#class_weight='balanced',  random_state=rep
    loss = log.fit(train_x, train_y)

    # with open('other_model/LR.pickle', 'wb') as f:
    #     pickle.dump(log, f)
    # with open('other_model/LR.pickle', 'rb') as f:
    #     log = pickle.load(f)
    pred_label = log.predict(test_x)
    pred_score = log.predict_proba(test_x)[:, 1]
    return pred_label, pred_score


def RFmodel(train_x, train_y, test_x, test_y):
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', class_weight='balanced') #
    # 拟合数据
    rf.fit(train_x, train_y)
    # 输出结果
    pred_test_score = rf.predict_proba(test_x)[:, 1]
    pred_label = rf.predict(test_x)
    print('RF acc_test:', (rf.score(test_x, test_y)))
    return pred_label, pred_test_score


def XGboostmodel(train_x, train_y, test_x, test_y):
    # scale_pos_weight: 正样本的权重，在二分类模型中，如果两个分类的样本比例失衡，
    # 可以设置该参数，模型效果会更好。比如，在研究疾病组和健康对照组分类时，postive：negative = 1:10，可以设置scale_pos_weight = 10，来平衡样本。
    xgb = XGBClassifier(n_estimators=1000,
                        booster='gbtree',
                        learning_rate=0.001,
                        max_depth=1,
                        subsample=0.7,
                        objective='binary:logistic',
                        eval_metric='auc'
                        ) # min_child_weight= 1,
    # 拟合数据

    xgb.fit(train_x, train_y) #, sample_weight=w_array
    # 输出结果
    pred_label = xgb.predict(test_x)
    pred_test_score = xgb.predict_proba(test_x)

    return pred_label, pred_test_score[:, 1]


def LightGBMmodel(train_x, train_y, test_x, test_y):
    lgb_train = lgb.Dataset(train_x, train_y)
    # lgb_eval = lgb.Dataset(test_x, test_y, reference=lgb_train)
    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbrt',
        'objective': 'binary',
        'metric': {'mse', 'auc'},  # 二进制对数损失
        'num_leaves':50,#50
        'max_depth':10,#10
        'min_data_in_leaf':20,#550/33
        'learning_rate': 0.02,
        'feature_fraction': 0.6,#0.9
        'bagging_fraction': 0.8,#0.8
        'bagging_freq': 5,
        'lambda_l1': 1,
        'lambda_l2': 0.1 # 越小l2正则程度越高
         # 'min_gain_to_split': 0.2
        # 'verbose': 5,
    }
    # train
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=500
                    ) # valid_sets=lgb_eval
    print('Start predicting...')
    pred_test_score = gbm.predict(test_x) # 输出的是概率结果
    return pred_test_score


def calculate_performance(true_label, predict_score, predict_label):
    cmp_one = np.ones_like(true_label)
    cmp_zero = np.zeros_like(true_label)
    fprs, tprs, thresholds = roc_curve(true_label, predict_score)
    auc_roc = roc_auc_score(true_label, predict_score)
    sen = np.sum(np.equal(true_label, cmp_one) & np.equal(predict_label, cmp_one)) / np.sum(
        np.equal(true_label, cmp_one))
    spe = np.sum(np.equal(true_label, cmp_zero) & np.equal(predict_label, cmp_zero)) / np.sum(
        np.equal(true_label, cmp_zero))
    acc = np.mean(np.equal(true_label, predict_label))

    precisions, recalls, ths = precision_recall_curve(true_label, predict_score)
    auc_prc = auc(recalls, precisions)
    recall = sen
    precision = np.sum(np.equal(true_label, cmp_one) & np.equal(predict_label, cmp_one)) / np.sum(np.equal(predict_label, cmp_one))
    f1 = 2 * precision * recall / (precision + recall)
    return auc_roc, sen, spe, acc, auc_prc, recall, precision, f1


def summary_results(groups_auc, groups_sen, groups_spe, groups_acc, groups_auc_prc, groups_recall, groups_precison, groups_f1):
    results = [np.mean(groups_auc, axis=0),
            np.mean(groups_sen,axis=0),
            np.mean(groups_spe,axis=0),
            np.mean(groups_acc,axis=0),
            np.mean(groups_auc_prc, axis=0),
            np.mean(groups_recall, axis=0),
            np.mean(groups_precison, axis=0),
            np.mean(groups_f1, axis=0),

            np.std(groups_auc,axis=0),
            np.std(groups_sen,axis=0),
            np.std(groups_spe,axis=0),
            np.std(groups_acc,axis=0),
            np.std(groups_auc_prc, axis=0),
            np.std(groups_recall, axis=0),
            np.std(groups_precison, axis=0),
            np.std(groups_f1, axis=0)
            ]
    return results


if __name__ == '__main__':
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    # 加载数据
    print('---load data---')
    data_all = pd.read_excel('data/data_norm_for_example2.xlsx')
    data_label = np.array(data_all)[:, 1]
    data_feature = np.array(data_all)[:, 2:]
    original_vars_size = data_feature.shape[1]
    print('---data shape:', data_feature.shape)
    algorithm = 'LSTM'  # ['LSTM', 'RF', 'LR', 'XGBoost', 'LightGBM']
    auc_roc_list = []
    sen_list = []
    spe_list = []
    acc_list = []
    auc_prc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    train_num = 0
    train_num_all = 5

    if algorithm == 'LSTM':
        for train_num in range(train_num_all):
            # 训练集测试集随机划分
            train_data, test_data, train_label, test_label = train_test_split(data_feature, data_label, test_size=0.3,
                                                                              stratify=data_label)
            print('======================train num:', train_num, algorithm)
            auc_roc, sen, spe, acc, auc_prc, recall, precision, f1 = LSTMmodel(train_data, train_label, test_data, test_label)
            auc_roc_list.append(auc_roc)
            sen_list.append(sen)
            spe_list.append(spe)
            acc_list.append(acc)
            auc_prc_list.append(auc_prc)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)

            results_list = summary_results(groups_auc=auc_roc_list,
                                           groups_sen=sen_list,
                                           groups_spe=spe_list,
                                           groups_acc=acc_list,
                                           groups_auc_prc=auc_prc_list,
                                           groups_recall=recall_list,
                                           groups_precison=precision_list,
                                           groups_f1=f1_list)
            print('-----------Finish!!!!! save!!!!')
            writer = pd.ExcelWriter('results_save/results_'+algorithm+'.xlsx')
            pd.DataFrame(results_list,
                         index=['mean_auc_roc', 'mean_sen', 'mean_spe', 'mean_acc', 'mean_auc_prc', 'mean_recall',
                                'mean_presicion', 'mean_f1', 'std_auc_roc', 'std_sen', 'std_spe', 'std_acc', 'std_auc_prc',
                                'std_recall',
                                'std_presicion', 'std_f1'],
                         ).to_excel(writer, sheet_name='test_metric')
            writer._save()
    else:
        for train_num in range(train_num_all):
            # 训练集测试集随机划分
            train_data, test_data, train_label, test_label = train_test_split(data_feature, data_label, test_size=0.3,
                                                                              stratify=data_label)
            print('======================train num:', train_num, algorithm)
            if algorithm == 'RF':
                pred_label, pred_score = RFmodel(train_data, train_label, test_data, test_label)
            elif algorithm == 'LR':
                pred_label, pred_score = LRmodel(train_data, train_label, test_data, test_label)
            elif algorithm == 'XGBoost':
                pred_label, pred_score = XGboostmodel(train_data, train_label, test_data, test_label)
            elif algorithm == 'LightGBM':
                pred_score = LightGBMmodel(train_data, train_label, test_data, test_label)
            else:
                print('no model...')
            fpr, tpr, thresholds = metrics.roc_curve(np.array(test_label), pred_score, pos_label=1)
            plt.plot(fpr, tpr)
            # plt.show()
            roc_auc = auc(fpr, tpr)  # 计算auc的值
            print('Model AUROC:', roc_auc)
            gmean = np.sqrt(tpr * (1 - fpr))
            # 查找最佳阈值
            index = np.argmax(gmean)
            thresholdOpt = round(thresholds[index], ndigits=4)
            gmeanOpt = round(gmean[index], ndigits=4)
            fprOpt = round(fpr[index], ndigits=4)
            tprOpt = round(tpr[index], ndigits=4)
            print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
            print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

            pred_test_label = (pred_score >= thresholdOpt)  # 将得分转化为对应的0&1值
            pred_test_label = [int(item) for item in pred_test_label]
            pred_test_label = np.array(pred_test_label).reshape([-1, 1])
            writer1 = pd.ExcelWriter('results_save/results_'+algorithm+'_pred.xlsx')
            pd.DataFrame(np.array(pred_score).reshape([-1, 1])).to_excel(writer1, sheet_name='pred_label_score')
            pd.DataFrame(np.array(pred_test_label).reshape([-1, 1])).to_excel(writer1, sheet_name='pred_label')
            writer1._save()
            auc_roc, sen, spe, acc, auc_prc, recall, precision, f1 = calculate_performance(
                np.array(test_label).squeeze(), pred_score.squeeze(), pred_test_label.squeeze())

            auc_roc_list.append(auc_roc)
            sen_list.append(sen)
            spe_list.append(spe)
            acc_list.append(acc)
            auc_prc_list.append(auc_prc)
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)

            results_list = summary_results(groups_auc=auc_roc_list,
                                           groups_sen=sen_list,
                                           groups_spe=spe_list,
                                           groups_acc=acc_list,
                                           groups_auc_prc=auc_prc_list,
                                           groups_recall=recall_list,
                                           groups_precison=precision_list,
                                           groups_f1=f1_list)
            print('-----------Finish!!!!! save!!!!')
            writer = pd.ExcelWriter('results_save/results_'+algorithm+'.xlsx')
            pd.DataFrame(results_list,
                         index=['mean_auc_roc', 'mean_sen', 'mean_spe', 'mean_acc', 'mean_auc_prc', 'mean_recall',
                                'mean_presicion', 'mean_f1', 'std_auc_roc', 'std_sen', 'std_spe', 'std_acc',
                                'std_auc_prc',
                                'std_recall',
                                'std_presicion', 'std_f1'],
                         ).to_excel(writer, sheet_name='test_metric')
            writer._save()












