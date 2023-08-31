import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
  
def shadow_a_plot(file='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val/image_shadow.txt', 
save_path='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val/image_shadow.png'):
    image = []
    N_shadow = []
    N_not_shadow = []
    N_all = []
    shadow_a = []
    # image N_shadow N_not_shadow N_all shadow_a

    for line in open(file, 'r'):
        lines = [i for i in line.split()]
        image.append(lines[0])
        N_shadow.append(int(lines[1]))
        N_not_shadow.append(int(lines[2]))
        N_all.append(int(lines[3]))
        shadow_a.append(float(lines[4]))
        
    plt.title("Shadow amount on ImageNet val. dataset")
    plt.xlabel('Image')
    plt.ylabel('Shadow_A')
    plt.yticks(shadow_a)
    plt.plot(image, shadow_a, marker = 'o', c = 'g')
    
    #plt.show()
    plt.savefig(save_path)
    print('done')

'''
Plot the shadow amount information for the  separate class
'''
def shadow_a_bins_plot(class_name):

    file='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/'+class_name+'/image_shadow.txt'
    save_path='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/'+class_name+'/image_shadow_bins10.png'
    no_shadow_info_path='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/'+class_name+'/no_shadow.txt'

    image = []
    N_shadow = []
    N_not_shadow = []
    N_all = []
    shadow_a = []
    no_shadow_a = 0
    # image N_shadow N_not_shadow N_all shadow_a

    for line in open(file, 'r'):
        lines = [i for i in line.split()]
        image.append(lines[0])
        N_shadow.append(int(lines[1]))
        N_not_shadow.append(int(lines[2]))
        N_all.append(int(lines[3]))
        shadow_a.append(float(lines[4]))
        if float(lines[4]) == 0:
            no_shadow_a += 1
            with open(no_shadow_info_path, 'a') as f:
                f.write(str(lines[0])+'\r\n')
    
    print('no_shadow_a', no_shadow_a) # 3198 images in ImageNet val. dataset do not have shadows at all

    plt.figure()    
    plt.title("Shadow amount on ImageNet val. dataset")
    plt.xlabel('Shadow Amount, %')
    plt.ylabel('Number of images')
    plt.hist(shadow_a, bins=10)

    plt.savefig(save_path)

    with open(no_shadow_info_path, 'a') as f:
        f.write(str(no_shadow_a)+'\r\n')

    print('done')

'''
Plot a graph with bins for shadow amount for all ImageNet val. dataset
'''
def shadow_a_bins_ImgNet_all_plot(args):
        
    image = []
    N_shadow = []
    N_not_shadow = []
    N_all = []
    shadow_a = []
    no_shadow_a = 0
    # image N_shadow N_not_shadow N_all shadow_a

    for line in open(args.image_shadow_txt, 'r'):
        lines = [i for i in line.split()]
        image.append(lines[0])
        N_shadow.append(int(lines[1]))
        N_not_shadow.append(int(lines[2]))
        N_all.append(int(lines[3]))
        shadow_a.append(float(lines[4]))
        if float(lines[4]) == 0:
            no_shadow_a += 1
            with open(args.save_no_shadow_info_path, 'a') as f:
                f.write(str(lines[0])+'\r\n')
    
    print('no_shadow_a', no_shadow_a) # images in ImageNet val. dataset do not have shadows at all

    plt.figure()
    plt.title("Shadow amount on ImageNet val. dataset")
    plt.xlabel('Shadow Amount, %')
    plt.ylabel('Number of images')
    plt.xticks(np.arange(0, 101, 5))
    plt.hist(shadow_a, bins=20)

    plt.savefig(args.save_plot_path)

    with open(args.save_no_shadow_info_path, 'a') as f:
        f.write(str(no_shadow_a)+'\r\n')

    print('Results saved: ', args.save_plot_path)
    print('done')

if __name__ == '__main__':
    #shadow_a_plot(file='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/n01440764/image_shadow.txt', 
    #              save_path='./models/model_SBU_EGNet_ablation/prediction_ImageNet-val-categories/n01440764/image_shadow.png')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_shadow_txt', type=str, default='./models/model_SBU_EGNet_ablation/prediction_ILSVRC2012_img_val/image_shadow.txt')
    parser.add_argument('--save_plot_path', type=str, default='./models/model_SBU_EGNet_ablation/prediction_ILSVRC2012_img_val/image_shadow_bins50.png')
    parser.add_argument('--save_no_shadow_info_path', type=str, default='./models/model_SBU_EGNet_ablation/prediction_ILSVRC2012_img_val/no_shadow.txt')
    FLAGS = parser.parse_args()

    shadow_a_bins_ImgNet_all_plot(FLAGS)