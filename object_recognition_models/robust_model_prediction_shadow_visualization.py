from robustbench.data import load_imagenet
from robustbench.utils import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import seaborn as sns
import pandas as pd


def read_shadow_amount_info(shadow_path):
    """
    Returns dict with image name and shadow amount on the image
    Input:
    shadow_path, string - path to the txt file with shadow amount imformation for each image in imagenet val. dataset
    
    Output:
    dict_img_shadow, dict - dict with image name (string) and shadow amount on the image (float)

    """
    print('Read shadow info...')
    dict_img_shadow = dict()
    for line in open(shadow_path, 'r'):
        lines = [i for i in line.split()]
        dict_img_shadow[lines[0]] = float(lines[4])
    return dict_img_shadow

def load_torch_all_test_data(x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """
    Returns torch x_test, y_test and paths_test for all imagenet val. dataset from saved files

    Input:
    x_test_path, y_test_path, paths_test_path, string - paths to the files

    Output:
    x_test, y_test, paths_test, torch

    """
    ### you can use the command 'load_imagenet' to load the max. 5000 test data samples
    #x_test, y_test, paths_test = load_imagenet(n_examples=n_examples)
    
    #torch.save(x_test, 'x_test.pt')
    #torch.save(y_test, 'y_test.pt')
    #torch.save(paths_test, 'paths_test.pt')
    
    print('Load data...')
    x_test = torch.load(x_test_path)
    y_test = torch.load(y_test_path)
    paths_test = torch.load(paths_test_path)
    print('Data loaded!')
    return x_test, y_test, paths_test

def robustbench_load_data(n_examples):
    """
    Returns n_examples of x_test, y_test, paths_test using robustbench data loading

    """
    print('Load data...')
    x_test, y_test, paths_test = load_imagenet(n_examples=n_examples)
    print('Data loaded!')
    return x_test, y_test, paths_test

def robustbench_model_predictions(model, dataset, threat_model, x_test):
    """
    Returns model's predictions and confidence of predictions using Softmax

    Input:
    model_name, string - name of the model from robustbench table
    dataset, string - name of the dataset (in our case 'imagenet')
    threat_model, string - type of corruptions in robustbench table ('Linf' in our case)

    Output:
    output, torch - model's predictions
    sm_output, torch - softmax model's predictions

    """
    print('Model predicts...')
    output = model(x_test)
    predicted_classes = output.max(1)[1]
    sm = torch.nn.Softmax(dim=1)
    sm_output = sm(output)
    print('Model Predictions...')
    return output, sm_output, predicted_classes

def class_info_plot(n_examples = 50, class_name = 'n07742313', shadow_path = './prediction_ImageNet-val-categories/...class_name...+/image_shadow.txt', 
                    model_name='Liu2023Comprehensive_Swin-L', dataset = 'imagenet', threat_model = 'Linf'):
    """
    Build a graph with predicted class probabilities and mismatches for a certain class in imagenet val. dataset
    (Before requires some tunig for the robustbench library: 
        1. Combine a .txt file with class and image names: n07742313/ILSVRC2012_val_00043445.JPEG, ...
        2. Upload the .txt file into robustbench/helper_files folder
        3. Change path in loaders.py file to the new .txt-file (line 70))
    
    Input:
    n_examples, int - number of samples for a certain class (max 50 for a val. dataset)
    class_name, string
    shadow_path, string - path to the txt file with shadow amount imformation for each image in imagenet val. dataset
    model_name, string - name of the model used from robustbench
    dataset, string - name of the dataset used for the robustbench
    
    Output:
    save_path, png - statistic visualization of prediction confidence for a certain class saved in a .png file
    
    """
    shadow_path = './prediction_ImageNet-val-categories/'+class_name+'/image_shadow.txt'

    dict_img_shadow = read_shadow_amount_info(shadow_path)
    x_test, y_test, paths_test = robustbench_load_data(n_examples)
    model = load_model(model_name=model_name, dataset=dataset, threat_model=threat_model)
    output, sm_output, predicted_classes = robustbench_model_predictions(model, dataset, threat_model, x_test)

    # model prediction statistics
    class_y_probability = []
    predicted_class_probability = []
    missmatch_x = []
    missmatch_y = []
    shadow_a = []
    shadow_no_x = []
    shadow_no_y = []
    for s in tqdm(range(n_examples)):
        class_y_probability.append(sm_output[s][y_test[s]].item()*100)
        predicted_class_probability.append(sm_output[s].max().item()*100)
        if predicted_classes[s] != y_test[s]:
            missmatch_x.append(s)
            missmatch_y.append(0)
        shadow_a.append(dict_img_shadow[paths_test[s].split('/')[-1]])
        if dict_img_shadow[paths_test[s].split('/')[-1]] == 0:
            shadow_no_x.append(s)
            shadow_no_y.append(0)

    acc = (output.max(1)[1] == y_test).float().sum()
    print('correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
    print('model accuracy: ', str(acc.item() / x_test.shape[0]))

    plt.figure()    
    plt.title("Shadow amount on ImageNet val. dataset class "+class_name)
    plt.xlabel('Image')
    plt.ylabel('Shadow Amount, %')

    plt.scatter(np.arange(n_examples), class_y_probability, label='Class Y probability')
    plt.scatter(np.arange(n_examples), predicted_class_probability, label='Predicted class probability')
    plt.plot(shadow_a, label='Shadow Amount')
    plt.scatter(missmatch_x, missmatch_y, marker='*', label='Misclassification')
    plt.legend(loc="upper left")

    save_path = './prediction_ImageNet-val-categories/'+class_name+'/shadow_prediction.png'
    plt.savefig(save_path)

def val_shadow_confidence_scatter_bar_sns(n_examples = 50000, shadow_path = './shadow_all_classes.txt', model_name = 'Liu2023Comprehensive_Swin-L', 
                                          dataset='imagenet', threat_model='Linf', 
                                          x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """
    Returns two .png images for correctly classiifed samples and misclassifications with prediction confidence (scatter plot) 
    and shadow amount and model's confidence distribution (bar plots)

    """
    dict_img_shadow = read_shadow_amount_info(shadow_path)
    x_test, y_test, paths_test = load_torch_all_test_data(x_test_path = x_test_path, y_test_path = y_test_path, paths_test_path = paths_test_path)
    print('Load model...')
    model = load_model(model_name = model_name, dataset = dataset, threat_model = threat_model)
    # model prediction statistics
    data_correct = []
    data_misclass = []
    acc = 0
    # model predictions were splitted as the size of the torch-tensor was too big
    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output, sm_output, predicted_classes = robustbench_model_predictions(model, dataset, threat_model, x_test[i:i+step])
        sm_output = sm_output.tolist()

        print('Collect statictics from predictions...')
        for s in range(i, i+step):
            if predicted_classes[s-i] != y_test[s]:
                data_misclass.append([sm_output[s-i][y_test[s]]*100, dict_img_shadow[paths_test[s].split('/')[-1]]])
            else:
                data_correct.append([sm_output[s-i][y_test[s]]*100, dict_img_shadow[paths_test[s].split('/')[-1]]])

        acc += (output.max(1)[1] == y_test[i:i+step]).float().sum().item()
        print('Correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
        print('Model accuracy: ', str(acc / x_test.shape[0]))
        del output
    
    df_misclass = pd.DataFrame(data_misclass, columns=['mismatch_x_pred_probability', 'mismatch_y_shadow_amount'])
    df_correct = pd.DataFrame(data_correct, columns=['correctly_class_x_pred_probability', 'correctly_class_y_shadow_amount'])

    sns_correct = sns.jointplot(data=df_correct, x="correctly_class_x_pred_probability", y="correctly_class_y_shadow_amount", marker="o")
    sns_correct.set_axis_labels("Prediction Confidence, %", "Shadow Amount, %")
    sns_correct.fig.suptitle("Correctly classified")
    save_path = './shadow_confidence_prediction_'+model_name+'_correct_test.png'
    sns_correct.figure.savefig(save_path)
    print('Data is savede here: ', save_path)
 
    sns_misclass = sns.jointplot(data=df_misclass, x="mismatch_x_pred_probability", y="mismatch_y_shadow_amount", color='orange', marker="o")
    sns_misclass.set_axis_labels("Prediction Confidence, %", "Shadow Amount, %")
    sns_misclass.fig.suptitle("Misclassifications")
    save_path = './shadow_confidence_prediction_'+model_name+'_mismatch_test.png'
    sns_misclass.figure.savefig(save_path)
    print('Data is savede here: ', save_path)

def val_shadow_confidence_plot(n_examples = 50000, shadow_path = './shadow_all_classes.txt', 
                               model_name = 'Liu2023Comprehensive_Swin-L', dataset='imagenet', threat_model='Linf',
                               x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """
    Returns two .png images for correctly classiifed samples and misclassifications with prediction confidence (scatter plot)

    """
    dict_img_shadow = read_shadow_amount_info(shadow_path)
    x_test, y_test, paths_test = load_torch_all_test_data(x_test_path = x_test_path, y_test_path = y_test_path, paths_test_path = paths_test_path)
    print('Load model...')
    model = load_model(model_name = model_name, dataset = dataset, threat_model = threat_model)

    # model prediction statistics
    mismatch_x_pred_probability = []
    mismatch_y_shadow_amount = []
    correctly_class_x_pred_probability = []
    correctly_class_y_shadow_amount = []
    acc = 0
    
    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output, sm_output, predicted_classes = robustbench_model_predictions(model, dataset, threat_model, x_test[i:i+step])
        sm_output = sm_output.tolist()

        print('Collect statictics from predictions...')
        for s in range(i, i+step):
            if predicted_classes[s-i] != y_test[s]:
                mismatch_x_pred_probability.append(sm_output[s-i][y_test[s]]*100)
                mismatch_y_shadow_amount.append(dict_img_shadow[paths_test[s].split('/')[-1]])
            else:
                correctly_class_x_pred_probability.append(sm_output[s-i][y_test[s]]*100)
                correctly_class_y_shadow_amount.append(dict_img_shadow[paths_test[s].split('/')[-1]])

        acc += (output.max(1)[1] == y_test[i:i+step]).float().sum().item()
        print('Correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
        print('Model accuracy: ', str(acc / x_test.shape[0]))
        del output

    plt.figure()    
    #plt.title("Correlation btw. shadow amount and prediction confidence on ImageNet")
    plt.title("Correctly classified")
    plt.xlabel('Prediction Confidence, %')
    plt.xticks(np.arange(0, 101, 20))
    plt.ylabel('Shadow Amount, %')

    plt.scatter(correctly_class_x_pred_probability, correctly_class_y_shadow_amount, label='Correctly classified')
    #plt.legend(loc="upper left")

    save_path = './shadow_confidence_prediction_'+model_name+'_correct_new.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

    plt.figure()    
    plt.title("Misclassification")
    plt.xlabel('Prediction Confidence, %')
    plt.xticks(np.arange(0, 101, 20))
    plt.ylabel('Shadow Amount, %')

    plt.scatter(mismatch_x_pred_probability, mismatch_y_shadow_amount, color='orange', label='Misclassification')
    #plt.legend(loc="upper left")

    save_path = './shadow_confidence_prediction_'+model_name+'_mismatch_new.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

def val_shadow_confidence_mean_count_one_model_plot(n_examples = 50000, shadow_path = './shadow_all_classes.txt', 
                                     model_name='Liu2023Comprehensive_Swin-L', dataset='imagenet', threat_model='Linf',
                                     x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """Returns two .png images:
    1. Correlation btw. shadow amount and number of correct/incorrect predictions on ImageNet for one model
    2. Correlation btw. shadow amount and mean prediction confidence of correct/incorrect predictions on ImageNet for one model

    """
    dict_img_shadow = read_shadow_amount_info(shadow_path)
    x_test, y_test, paths_test = load_torch_all_test_data(x_test_path = x_test_path, y_test_path = y_test_path, paths_test_path = paths_test_path)
    print('Load model...')
    model = load_model(model_name, dataset, threat_model)

    # model prediction statistics
    mismatch_shadow_step_1_count = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_mean = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_sum = np.zeros(101)
    correct_shadow_step_1_count = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_mean = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_sum = np.zeros(101)
    acc = 0

    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output, sm_output, predicted_classes = robustbench_model_predictions(model, dataset, threat_model, x_test[i:i+step])
        sm_output = sm_output.tolist()

        print('Collect statictics from predictions...')
        for s in range(i, i+step):
            shadow_a = round(dict_img_shadow[paths_test[s].split('/')[-1]])

            if predicted_classes[s-i] != y_test[s]:
                mismatch_shadow_step_1_count[shadow_a] += 1
                mismatch_shadow_step_1_conf_pred_sum[shadow_a] += sm_output[s-i][y_test[s]]*100
                mismatch_shadow_step_1_conf_pred_mean[shadow_a] = mismatch_shadow_step_1_conf_pred_sum[shadow_a] / mismatch_shadow_step_1_count[shadow_a]
            else:
                correct_shadow_step_1_count[shadow_a]+=1
                correct_shadow_step_1_count_conf_pred_sum[shadow_a] += sm_output[s-i][y_test[s]]*100
                correct_shadow_step_1_count_conf_pred_mean[shadow_a] = correct_shadow_step_1_count_conf_pred_sum[shadow_a] / correct_shadow_step_1_count[shadow_a]

        acc += (output.max(1)[1] == y_test[i:i+step]).float().sum().item()
        print('Correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
        print('Model accuracy: ', str(acc / x_test.shape[0]))
        del output

    plt.figure()    
    plt.title("Correlation btw. shadow amount and number of correct/incorrect predictions on ImageNet")
    plt.xlabel('Number of predictions')
    plt.yticks(np.arange(0, 101, 5))
    plt.xticks(np.arange(0, 10000, 1000))
    plt.ylabel('Shadow Amount, %')

    plt.hist([correct_shadow_step_1_count, mismatch_shadow_step_1_count], label=['Correctly classified', 'Misclassification'])
    #plt.hist(mismatch_shadow_step_1_count, bins=50, histtype='step', stacked=True, fill=False, label='Misclassification')
    plt.legend(loc="upper left")

    save_path = './shadow_confidence_count_prediction_no_bins_'+model_name+'_two_graphs_on_one.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

    
    plt.figure()    
    plt.title("Correlation btw. shadow amount and number of correct/incorrect predictions on ImageNet")
    plt.ylabel('Number of predictions')
    plt.xticks(np.arange(0, 101, 5))
    plt.xlabel('Shadow Amount, %')

    plt.plot(correct_shadow_step_1_count, label='Correctly classified')
    plt.plot(mismatch_shadow_step_1_count, label='Misclassification')
    plt.legend(loc="upper left")

    save_path = './shadow_confidence_count_prediction_line_'+model_name+'_new.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

    plt.figure()    
    plt.title("Correlation btw. shadow amount and mean prediction confidence of correct/incorrect predictions on ImageNet")
    plt.xlabel('Shadow Amount, %')
    plt.xticks(np.arange(0, 101, 5))
    plt.ylabel('Mean Prediction Confidence, %')

    plt.plot(correct_shadow_step_1_count_conf_pred_mean, label='Correctly classified')
    plt.plot(mismatch_shadow_step_1_conf_pred_mean, label='Misclassification')
    plt.legend(loc="upper left")

    save_path = './shadow_confidence_mean_prediction_'+model_name+'_new.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

def val_shadow_confidence_mean_count_two_models_plot(n_examples = 50000, shadow_path = './shadow_all_classes.txt', 
                                    model_name1='Liu2023Comprehensive_Swin-L', model_name2='Salman2020Do_R50', dataset='imagenet', threat_model='Linf',
                                    x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """Returns two .png images:
    1. Correlation btw. shadow amount and number of correct/incorrect predictions on ImageNet for two models
    2. Correlation btw. shadow amount and mean prediction confidence of correct/incorrect predictions on ImageNet for two models

    """

    mismatch_shadow_step_1_count_m1, mismatch_shadow_step_1_conf_pred_mean_m1, mismatch_shadow_step_1_conf_pred_sum_m1, correct_shadow_step_1_count_m1, correct_shadow_step_1_count_conf_pred_mean_m1, correct_shadow_step_1_count_conf_pred_sum_m1 = val_shadow_confidence_count(n_examples, shadow_path, model_name1, dataset, threat_model, x_test_path, y_test_path, paths_test_path)
    mismatch_shadow_step_1_count_m2, mismatch_shadow_step_1_conf_pred_mean_m2, mismatch_shadow_step_1_conf_pred_sum_m2, correct_shadow_step_1_count_m2, correct_shadow_step_1_count_conf_pred_mean_m2, correct_shadow_step_1_count_conf_pred_sum_m2 = val_shadow_confidence_count(n_examples, shadow_path, model_name2, dataset, threat_model, x_test_path, y_test_path, paths_test_path)
    
    """plt.figure()    
    plt.title("Correlation btw. shadow amount and number of correct/incorrect predictions on ImageNet")
    plt.xlabel('Number of predictions')
    plt.yticks(np.arange(0, 101, 5))
    plt.xticks(np.arange(0, 10000, 1000))
    plt.ylabel('Shadow Amount, %')

    plt.hist([correct_shadow_step_1_count_m1, mismatch_shadow_step_1_count_m1, correct_shadow_step_1_count_m2, mismatch_shadow_step_1_count_m2], 
    label=['Correctly classified ' + model_name1, 'Misclassification ' + model_name1, 'Correctly classified ' + model_name2, 'Misclassification ' + model_name2])
    #plt.hist(mismatch_shadow_step_1_count, bins=50, histtype='step', stacked=True, fill=False, label='Misclassification')
    plt.legend(loc="upper left")

    save_path = './shadow_confidence_count_prediction_no_bins_'+model_name1+'_'+model_name2+'_two_graphs_on_one.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)"""

    
    plt.figure()    
    plt.title("Correlation btw. shadow amount and number of correct/incorrect predictions on ImageNet")
    plt.ylabel('Number of predictions')
    plt.xticks(np.arange(0, 101, 5))
    plt.xlabel('Shadow Amount, %')

    plt.plot(correct_shadow_step_1_count_m1, label='Correctly classified '+model_name1)
    plt.plot(mismatch_shadow_step_1_count_m1, label='Misclassification '+model_name1)
    plt.plot(correct_shadow_step_1_count_m2, label='Correctly classified '+model_name2)
    plt.plot(mismatch_shadow_step_1_count_m2, label='Misclassification '+model_name2)
    plt.legend(loc="upper left")

    save_path = './shadow_confidence_count_prediction_line_'+model_name1+'_'+model_name2+'_new.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

    plt.figure()    
    plt.title("Correlation btw. shadow amount and mean prediction confidence of correct/incorrect predictions on ImageNet")
    plt.xlabel('Shadow Amount, %')
    plt.xticks(np.arange(0, 101, 5))
    plt.ylabel('Mean Prediction Confidence, %')

    plt.plot(correct_shadow_step_1_count_conf_pred_mean_m1, label='Correctly classified '+model_name1)
    plt.plot(mismatch_shadow_step_1_conf_pred_mean_m1, label='Misclassification '+model_name1)
    plt.plot(correct_shadow_step_1_count_conf_pred_mean_m2, label='Correctly classified '+model_name2)
    plt.plot(mismatch_shadow_step_1_conf_pred_mean_m2, label='Misclassification '+model_name2)
    plt.legend(loc="upper left")

    save_path = './shadow_confidence_mean_prediction_'+model_name1+'_'+model_name2+'_new.png'
    plt.savefig(save_path)
    print('Data is savede here: ', save_path)

def val_shadow_confidence_count(n_examples = 50000, shadow_path = './shadow_all_classes.txt', 
                                model_name='Liu2023Comprehensive_Swin-L', dataset='imagenet', threat_model='Linf',
                                x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """
    Returns 
    1. number of misclassifications and correct classifications, 
    2. means of model's prediction confidence of misclassifications and correct classifications,
    3. sums of model's prediction confidence of misclassifications and correct classifications,

    """
    dict_img_shadow = read_shadow_amount_info(shadow_path)
    x_test, y_test, paths_test = load_torch_all_test_data(x_test_path = x_test_path, y_test_path = y_test_path, paths_test_path = paths_test_path)
    print('Load model...')
    model = load_model(model_name = model_name, dataset = dataset, threat_model = threat_model)
    # model prediction statistics
    mismatch_shadow_step_1_count = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_mean = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_sum = np.zeros(101)
    correct_shadow_step_1_count = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_mean = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_sum = np.zeros(101)
    acc = 0

    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output, sm_output, predicted_classes = robustbench_model_predictions(model, dataset, threat_model, x_test[i:i+step])
        sm_output = sm_output.tolist()

        print('Collect statictics from predictions...')
        for s in range(i, i+step):
            shadow_a = round(dict_img_shadow[paths_test[s].split('/')[-1]])

            if predicted_classes[s-i] != y_test[s]:
                mismatch_shadow_step_1_count[shadow_a] += 1
                mismatch_shadow_step_1_conf_pred_sum[shadow_a] += sm_output[s-i][y_test[s]]*100
                mismatch_shadow_step_1_conf_pred_mean[shadow_a] = mismatch_shadow_step_1_conf_pred_sum[shadow_a] / mismatch_shadow_step_1_count[shadow_a]
            else:
                correct_shadow_step_1_count[shadow_a]+=1
                correct_shadow_step_1_count_conf_pred_sum[shadow_a] += sm_output[s-i][y_test[s]]*100
                correct_shadow_step_1_count_conf_pred_mean[shadow_a] = correct_shadow_step_1_count_conf_pred_sum[shadow_a] / correct_shadow_step_1_count[shadow_a]

        acc += (output.max(1)[1] == y_test[i:i+step]).float().sum().item()
        print('Correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
        print('Model accuracy: ', str(acc / x_test.shape[0]))
        del output

    return mismatch_shadow_step_1_count, mismatch_shadow_step_1_conf_pred_mean, mismatch_shadow_step_1_conf_pred_sum, correct_shadow_step_1_count, correct_shadow_step_1_count_conf_pred_mean, correct_shadow_step_1_count_conf_pred_sum

def val_shadow_confidence(n_examples = 50000, shadow_path = './shadow_all_classes.txt', 
                          model_name='Liu2023Comprehensive_Swin-L', dataset='imagenet', threat_model='Linf',
                          x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """
    Returns 
    1. number of misclassifications and correct classifications, 
    2. means of model's prediction confidence of misclassifications and correct classifications,
    3. sums of model's prediction confidence of misclassifications and correct classifications,

    """
    dict_img_shadow = read_shadow_amount_info(shadow_path)
    x_test, y_test, paths_test = load_torch_all_test_data(x_test_path = x_test_path, y_test_path = y_test_path, paths_test_path = paths_test_path)
    print('Load model...')
    model = load_model(model_name = model_name, dataset = dataset, threat_model = threat_model)
    # model prediction statistics
    misclass = []
    correct = []
    acc = 0

    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output, sm_output, predicted_classes = robustbench_model_predictions(model, dataset, threat_model, x_test[i:i+step])
        sm_output = sm_output.tolist()

        print('Collect statictics from predictions...')
        for s in range(i, i+step):
            shadow_a = dict_img_shadow[paths_test[s].split('/')[-1]]

            if predicted_classes[s-i] != y_test[s]:
                misclass.append([shadow_a, sm_output[s-i][y_test[s]]*100])
            else:
                correct.append([shadow_a, sm_output[s-i][y_test[s]]*100])

        acc += (output.max(1)[1] == y_test[i:i+step]).float().sum().item()
        print('Correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
        print('Model accuracy: ', str(acc / x_test.shape[0]))
        del output

    return misclass, correct

def val_shadow_confidence_hist_plot(n_examples = 50000, shadow_path = './shadow_all_classes.txt', 
                                    model_name='Liu2023Comprehensive_Swin-L', dataset='imagenet', threat_model='Linf',
                                    x_test_path = 'x_test.pt', y_test_path = 'y_test.pt', paths_test_path = 'paths_test.pt'):
    """
    Returns correlation btw. shadow amount and mean prediction confidence of correct/incorrect predictions on ImageNet

    """

    misclass1, correct1 = val_shadow_confidence(n_examples, shadow_path, model_name, dataset, threat_model, x_test_path, y_test_path, paths_test_path)
    df1_misclass = pd.DataFrame(misclass1, columns=["shadow_a", "confidence"])
    df1_correct = pd.DataFrame(correct1, columns=["shadow_a", "confidence"])

    fig, axes = plt.subplots(1, 2)
    axes[0].set(xlim =(0, 100), ylim =(0, 100))
    axes[1].set(xlim =(0, 100), ylim =(0, 100))
    fig.suptitle(model_name)
    sns.histplot(df1_correct, x="confidence", y="shadow_a", ax = axes[0], legend=False, bins=20, cbar=True, cbar_kws=dict(shrink=.75), pthresh=.05, pmax=.9)
    axes[0].set_title("Correct classified")
    axes[0].set_xlabel('Prediction Confidence, %')
    axes[0].set_ylabel('Shadow Amount, %')
    sns.histplot(df1_misclass, x="confidence", y="shadow_a", ax = axes[1], legend=False, bins=20, cbar=True, color = "orange", cbar_kws=dict(shrink=.75), pthresh=.05, pmax=.9)
    axes[1].set_title("Misclassification")
    axes[1].set_xlabel('Prediction Confidence, %')
    axes[1].set_ylabel(' ')

    save_path = './shadow_confidence_mean_'+model_name+'_all_test.png'
    fig.savefig(save_path)
    print('Data is saved here: ', save_path)


def all_shadow_info_txt(folder='./prediction_ImageNet-val-categories/'):
    """
    Returns .txt file with shadow info for all classes in imagenet val. dataset
    
    """
    subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    list_shadow_all_path = './shadow_all_classes.txt'

    with open(list_shadow_all_path, 'a') as f:
        for subf in subfolders:
            class_name = subf
            shadow_path = folder + class_name + '/image_shadow.txt'

            # form image-shadow txt file
            for line in open(shadow_path, 'r'):
                f.write(line)

if __name__ == '__main__':
    n_examples = 50000
    shadow_path = './shadow_all_classes.txt'
    model_name = 'Salman2020Do_R50'
    dataset='imagenet'
    threat_model='Linf'
    x_test_path = 'x_test.pt'
    y_test_path = 'y_test.pt'
    paths_test_path = 'paths_test.pt'

    """all_shadow_info_txt()"""
    """val_shadow_confidence_plot(n_examples, shadow_path, 
                               model_name, dataset, threat_model,
                               x_test_path, y_test_path, paths_test_path)"""
    """val_shadow_confidence_scatter_bar_sns(n_examples, shadow_path, model_name, 
                                          dataset, threat_model, 
                                          x_test_path, y_test_path, paths_test_path)"""
    """val_shadow_confidence_mean_count_one_model_plot(n_examples, shadow_path, model_name, dataset, threat_model, x_test_path, y_test_path, paths_test_path)"""
    """val_shadow_confidence_mean_count_two_models_plot(n_examples, shadow_path, 'Liu2023Comprehensive_Swin-L', 'Salman2020Do_R50', dataset, threat_model, x_test_path, y_test_path, paths_test_path)"""
    val_shadow_confidence_hist_plot(n_examples, shadow_path, model_name, dataset, threat_model, x_test_path, y_test_path, paths_test_path)
