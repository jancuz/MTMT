from robustbench.data import load_imagenet
from robustbench.utils import load_model
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

def class_corr_plot():
    n_examples = 50
    class_name = 'n07742313'
    shadow_path = './prediction_ImageNet-val-categories/'+class_name+'/image_shadow.txt'


    # form image-shadow dict
    dict_img_shadow = dict()
    for line in open(shadow_path, 'r'):
        lines = [i for i in line.split()]
        dict_img_shadow[lines[0]] = float(lines[4])

    # load test data from one class
    x_test, y_test, paths_test = load_imagenet(n_examples=n_examples)
    # load model
    model = load_model(model_name='Liu2023Comprehensive_Swin-L', dataset='imagenet', threat_model='Linf')
    # model predictions
    output = model(x_test)
    predicted_classes = output.max(1)[1]

    #print(model)
    sm = torch.nn.Softmax(dim=1)
    sm_output = sm(output)

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
        #print('class_y_probability: ', str(sm_output[s][y_test[s]].item()*100))
        predicted_class_probability.append(sm_output[s].max().item()*100)
        #print('predicted_class_probability: ', str(sm_output[s].max().item()*100))
        if predicted_classes[s] != y_test[s]:
            missmatch_x.append(s)
            missmatch_y.append(0)
        shadow_a.append(dict_img_shadow[paths_test[s].split('/')[-1]])
        if dict_img_shadow[paths_test[s].split('/')[-1]] == 0:
            shadow_no_x.append(s)
            shadow_no_y.append(0)
        #print('shadow_a: ', str(dict_img_shadow[paths_test[s].split('/')[-1]]))

    acc = (output.max(1)[1] == y_test).float().sum()
    print('correctly predicted from ' + str(n_examples) + ' images: ', str(acc))
    print('model accuracy: ', str(acc.item() / x_test.shape[0]))

    plt.figure()    
    plt.title("Shadow amount on ImageNet val. dataset class "+class_name)
    plt.xlabel('Image')
    #plt.xticks(np.arange(1, n_examples+1))
    plt.ylabel('Shadow Amount, %')

    plt.scatter(np.arange(n_examples), class_y_probability, label='Class Y probability')
    plt.scatter(np.arange(n_examples), predicted_class_probability, label='Predicted class probability')
    plt.plot(shadow_a, label='Shadow Amount')
    plt.scatter(missmatch_x, missmatch_y, marker='*', label='Misclassification')
    plt.legend(loc="upper left")

    save_path = './prediction_ImageNet-val-categories/'+class_name+'/shadow_prediction.png'
    plt.savefig(save_path)

def val_shadow_confidence_plot(shadow_path, model_name='Liu2023Comprehensive_Swin-L'):
    n_examples = 50000
    dict_img_shadow = dict()

    # write shadow info for each class into one dict
    # form image-shadow dict
    print('Read shadow info...')
    for line in open(shadow_path, 'r'):
        lines = [i for i in line.split()]
        dict_img_shadow[lines[0]] = float(lines[4])

    print('Load data...')
    
    ### you can use the command 'load_imagenet' to load the max. 5000 test data samples
    #x_test, y_test, paths_test = load_imagenet(n_examples=n_examples)
    
    #torch.save(x_test, 'x_test.pt')
    #torch.save(y_test, 'y_test.pt')
    #torch.save(paths_test, 'paths_test.pt')
    
    # load all test data
    x_test = torch.load('x_test.pt')
    y_test = torch.load('y_test.pt')
    paths_test = torch.load('paths_test.pt')
    print('Data loaded!')
    print('Load model...')
    # load model
    model = load_model(model_name=model_name, dataset='imagenet', threat_model='Linf')

    # model prediction statistics
    mismatch_x_pred_probability = []
    mismatch_y_shadow_amount = []
    correctly_class_x_pred_probability = []
    correctly_class_y_shadow_amount = []
    acc = 0
    #output = torch.zeros(n_examples, 1000)
    
    # model predictions
    print('Model predicts...')
    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output = model(x_test[i:i+step])

        print('Model Predictions...')
        predicted_classes = output.max(1)[1]

        #print(model)
        sm = torch.nn.Softmax(dim=1)
        sm_output = sm(output).tolist()

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

def val_shadow_confidence_count_plot(shadow_path, model_name='Liu2023Comprehensive_Swin-L'):
    n_examples = 50000
    dict_img_shadow = dict()

    # write shadow info for each class into one dict
    # form image-shadow dict
    print('Read shadow info...')
    for line in open(shadow_path, 'r'):
        lines = [i for i in line.split()]
        dict_img_shadow[lines[0]] = float(lines[4])

    print('Load data...')
    
    ### you can use the command 'load_imagenet' to load the max. 5000 test data samples
    #x_test, y_test, paths_test = load_imagenet(n_examples=n_examples)
    
    #torch.save(x_test, 'x_test.pt')
    #torch.save(y_test, 'y_test.pt')
    #torch.save(paths_test, 'paths_test.pt')
    
    # load all test data
    x_test = torch.load('x_test.pt')
    y_test = torch.load('y_test.pt')
    paths_test = torch.load('paths_test.pt')
    print('Data loaded!')
    print('Load model...')
    # load model
    model = load_model(model_name=model_name, dataset='imagenet', threat_model='Linf')

    # model prediction statistics
    mismatch_shadow_step_1_count = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_mean = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_sum = np.zeros(101)
    correct_shadow_step_1_count = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_mean = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_sum = np.zeros(101)
    acc = 0
    #output = torch.zeros(n_examples, 1000)
    
    # model predictions
    print('Model predicts...')
    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output = model(x_test[i:i+step])

        print('Model Predictions...')
        predicted_classes = output.max(1)[1]

        #print(model)
        sm = torch.nn.Softmax(dim=1)
        sm_output = sm(output).tolist()

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

def val_shadow_confidence_count(shadow_path, model_name='Liu2023Comprehensive_Swin-L'):
    n_examples = 50000
    dict_img_shadow = dict()

    # write shadow info for each class into one dict
    # form image-shadow dict
    print('Read shadow info...')
    for line in open(shadow_path, 'r'):
        lines = [i for i in line.split()]
        dict_img_shadow[lines[0]] = float(lines[4])

    print('Load data...')
    
    ### you can use the command 'load_imagenet' to load the max. 5000 test data samples
    #x_test, y_test, paths_test = load_imagenet(n_examples=n_examples)
    
    #torch.save(x_test, 'x_test.pt')
    #torch.save(y_test, 'y_test.pt')
    #torch.save(paths_test, 'paths_test.pt')
    
    # load all test data
    x_test = torch.load('x_test.pt')
    y_test = torch.load('y_test.pt')
    paths_test = torch.load('paths_test.pt')
    print('Data loaded!')
    print('Load model...')
    # load model
    model = load_model(model_name=model_name, dataset='imagenet', threat_model='Linf')

    # model prediction statistics
    mismatch_shadow_step_1_count = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_mean = np.zeros(101)
    mismatch_shadow_step_1_conf_pred_sum = np.zeros(101)
    correct_shadow_step_1_count = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_mean = np.zeros(101)
    correct_shadow_step_1_count_conf_pred_sum = np.zeros(101)
    acc = 0
    #output = torch.zeros(n_examples, 1000)
    
    # model predictions
    print('Model predicts...')
    step = 100
    for i in tqdm(range(0, n_examples, step)):
        output = model(x_test[i:i+step])

        print('Model Predictions...')
        predicted_classes = output.max(1)[1]

        #print(model)
        sm = torch.nn.Softmax(dim=1)
        sm_output = sm(output).tolist()

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

def val_shadow_confidence_count_plot(shadow_path, model_name1='Liu2023Comprehensive_Swin-L', model_name2='Salman2020Do_R50'):
    mismatch_shadow_step_1_count_m1, mismatch_shadow_step_1_conf_pred_mean_m1, mismatch_shadow_step_1_conf_pred_sum_m1, correct_shadow_step_1_count_m1, correct_shadow_step_1_count_conf_pred_mean_m1, correct_shadow_step_1_count_conf_pred_sum_m1 = val_shadow_confidence_count(shadow_path, model_name1)
    mismatch_shadow_step_1_count_m2, mismatch_shadow_step_1_conf_pred_mean_m2, mismatch_shadow_step_1_conf_pred_sum_m2, correct_shadow_step_1_count_m2, correct_shadow_step_1_count_conf_pred_mean_m2, correct_shadow_step_1_count_conf_pred_sum_m2 = val_shadow_confidence_count(shadow_path, model_name2)
    
    plt.figure()    
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
    print('Data is savede here: ', save_path)

    
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

def all_shadow_info_txt(folder='./prediction_ImageNet-val-categories/'):
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
    #all_shadow_info_txt()
    #val_shadow_confidence_plot(shadow_path='./shadow_all_classes.txt', model_name='Salman2020Do_R50')
    #val_shadow_confidence_count_plot(shadow_path='./shadow_all_classes.txt', model_name='Salman2020Do_R50')
    val_shadow_confidence_count_plot(shadow_path='./shadow_all_classes.txt', model_name1='Liu2023Comprehensive_Swin-L', model_name2='Salman2020Do_R50')
