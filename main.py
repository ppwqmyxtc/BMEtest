import model
import streamlit as st
import pandas as pd
import numpy as np
from bokeh.plotting import figure
import time



if __name__ == '__main__':
    
    with st.sidebar:
        
        algorithm = st.selectbox('Select the Training Model', ['LSTM', 'RF', 'LR', 'XGBoost', 'LightGBM'], index=0)
        
    if 'algorithm' not in st.session_state:
        st.session_state.algorithm = algorithm
    
    upload_file = st.file_uploader("Upload your data, data type is xlsx", type=["xlsx"])
    
    if upload_file is not None:
        
        if ('upload_file' not in st.session_state or st.session_state.upload_file != upload_file) or st.session_state.algorithm != algorithm:
            
            st.session_state.upload_file = upload_file
            st.secrets.algorithm  = algorithm 
            
            input_data = pd.read_excel(upload_file)
            data_label = np.array(input_data)[:, 1]
            data_feature = np.array(input_data)[:, 2:]
            
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
                
                # make a progress bar
                progress_bar = st.progress(0, text='Training...')

                for train_num in range(train_num_all):
                    
                    progress_bar.progress(train_num / train_num_all, text='Training...')
                    
                    # 训练集测试集随机划分
                    train_data, test_data, train_label, test_label = model.train_test_split(data_feature, data_label, test_size=0.3,
                                                                                        stratify=data_label)
                    # print('======================train num:', train_num, algorithm)
                    auc_roc, sen, spe, acc, auc_prc, recall, precision, f1 = model.LSTMmodel(train_data, train_label, test_data, test_label)
                    auc_roc_list.append(auc_roc)
                    sen_list.append(sen)
                    spe_list.append(spe)
                    acc_list.append(acc)
                    auc_prc_list.append(auc_prc)
                    recall_list.append(recall)
                    precision_list.append(precision)
                    f1_list.append(f1)
                
                progress_bar.progress(100, text='Training Done!')
                

                results_list = model.summary_results(groups_auc=auc_roc_list,
                                                groups_sen=sen_list,
                                                groups_spe=spe_list,
                                                groups_acc=acc_list,
                                                groups_auc_prc=auc_prc_list,
                                                groups_recall=recall_list,
                                                groups_precison=precision_list,
                                                groups_f1=f1_list)
                
                results_df = pd.DataFrame(results_list,index=['mean_auc_roc', 'mean_sen', 'mean_spe', 'mean_acc', 'mean_auc_prc', 'mean_recall',
                                    'mean_presicion', 'mean_f1', 'std_auc_roc', 'std_sen', 'std_spe', 'std_acc',
                                    'std_auc_prc',
                                    'std_recall',
                                    'std_presicion', 'std_f1'],
                                        )
                
                # show the results df
                st.dataframe(results_df)
            
            else:
                
                progress_bar = st.progress(0, text='Training...')
                
                for train_num in range(train_num_all):
                    
                    progress_bar.progress(train_num / train_num_all, text='Training...')
                    
                    train_data, test_data, train_label, test_label = model.train_test_split(data_feature, data_label, test_size=0.3,
                                                                                    stratify=data_label)
                    if algorithm == 'RF':
                        pred_label, pred_score = model.RFmodel(train_data, train_label, test_data, test_label)
                    elif algorithm == 'LR':
                        pred_label, pred_score = model.LRmodel(train_data, train_label, test_data, test_label)
                    elif algorithm == 'XGBoost':
                        pred_label, pred_score = model.XGboostmodel(train_data, train_label, test_data, test_label)
                    elif algorithm == 'LightGBM':
                        pred_score = model.LightGBMmodel(train_data, train_label, test_data, test_label)
                    else:
                        print('no model...')
                    fpr, tpr, thresholds = model.metrics.roc_curve(np.array(test_label), pred_score, pos_label=1)
                    # plt.show()
                    roc_auc = model.auc(fpr, tpr)  # 计算auc的值
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
                    auc_roc, sen, spe, acc, auc_prc, recall, precision, f1 = model.calculate_performance(
                        np.array(test_label).squeeze(), pred_score.squeeze(), pred_test_label.squeeze())

                    auc_roc_list.append(auc_roc)
                    sen_list.append(sen)
                    spe_list.append(spe)
                    acc_list.append(acc)
                    auc_prc_list.append(auc_prc)
                    recall_list.append(recall)
                    precision_list.append(precision)
                    f1_list.append(f1)
                    
                progress_bar.progress(100, text='Training Done!')

                results_list = model.summary_results(groups_auc=auc_roc_list,
                                            groups_sen=sen_list,
                                            groups_spe=spe_list,
                                            groups_acc=acc_list,
                                            groups_auc_prc=auc_prc_list,
                                            groups_recall=recall_list,
                                            groups_precison=precision_list,
                                            groups_f1=f1_list)

                results_df = pd.DataFrame(results_list,
                            index=['mean_auc_roc', 'mean_sen', 'mean_spe', 'mean_acc', 'mean_auc_prc', 'mean_recall',
                                    'mean_presicion', 'mean_f1', 'std_auc_roc', 'std_sen', 'std_spe', 'std_acc',
                                    'std_auc_prc',
                                    'std_recall',
                                    'std_presicion', 'std_f1'],
                            )
                
                st.dataframe(results_df)
            
            
            