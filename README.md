# A Multi-task Mean Teacher for Semi-supervised Shadow Detection

by Zhihao Chen, Lei Zhu, Liang Wan, Song Wang, Wei Feng, and Pheng-Ann Heng [[paper link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_A_Multi-Task_Mean_Teacher_for_Semi-Supervised_Shadow_Detection_CVPR_2020_paper.pdf)]
forked from [original repository](https://github.com/eraserNut/MTMT)

***

## Citation
@inproceedings{chen20MTMT,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Chen, Zhihao and Zhu, Lei and Wan, Liang and Wang, Song and Feng, Wei and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {A Multi-task Mean Teacher for Semi-supervised Shadow Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {CVPR},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2020}    
}

## Our project - Role of shadows in object recognition task
* [Main repository](https://github.com/jancuz/ShadowProject.git)
* Shadow amount calculation and visualization
* Correlation analysis btw. shadow amount and prediction confidence of the robust models

## Trained Model (from [original repository](https://github.com/eraserNut/MTMT))
You can download the trained model which is reported in our paper at [BaiduNetdisk](https://pan.baidu.com/s/1yjnsjE7mDPnEaHxdtNFhhQ)(password: h52i) or [Google Drive](https://drive.google.com/file/d/1s-4BSmz9j8u2_WoUnzNYL0QjRYFEeEkU/view?usp=share_link).

## Requirement
See [requirements.txt](https://github.com/jancuz/MTMT/blob/master/requirements.txt)
* Python 3.6
* PyTorch 1.3.1(After 0.4.0 would be ok)
* torchvision
* numpy
* tqdm
* PIL
* pydensecrf ([here](https://github.com/Andrew-Qibin/dss_crf) to install)
* ...

## Training and Testing + Useful links
See [original repository](https://github.com/eraserNut/MTMT)

## Shadow amount calculation and visualization
To calculate the shadow amount with MTMT-net on ImageNet val. dataset:
* Place ImageNet val. dataset into the folder ```data```
* Download the MTMT-net model from the [Google Drive](https://drive.google.com/file/d/1s-4BSmz9j8u2_WoUnzNYL0QjRYFEeEkU/view?usp=share_link) and place it into the folder ```models```
* Run the ```test_MT.py``` with the specified path to the data and model:

```
python test_MT.py --root_path "path_to_the_data" --snapshot_path "path_to_the_model" --epoch_name "epoch_name" --dataset_name "ImageNet_val" --shadow_count True
```
* to save the output of the model (shadow-detected masks) use the following command:
```
python test_MT.py --root_path "path_to_the_data" --snapshot_path "path_to_the_model" --epoch_name "epoch_name" --dataset_name "ImageNet_val" --shadow_count True --save_result True --test_save_path "path_to_save_model_output"
```
To visualize the shadow amount with [MTMT-net](https://github.com/jancuz/MTMT.git) on ImageNet val. dataset:
* Calculate the shadow amount using the previous steps or download the file ```image_shadow.txt``` from the [google-disk]()
* Run the ```shadow_amount_visualization.py``` with the specified path to the data:

```
python shadow_amount_visualization.py --image_shadow_txt "path_to_the_image_shadow_txt_file" --save_plot_path "path_to_the_graph" --save_no_shadow_info_path "path_to_txt_file_to_save_no_shadow_images_names"
```  

For the ImageNet val. dataset shadow amount was calculated and visualized:
<p align="center"><img src="imgs/shadow_ammount_ImageNet.png" width="400">
 
* ImageNet val. dataset consists of 50.000 images
* 7.574 images (0,15%) in ImageNet val. dataset do not have shadows at all

## Correlation analysis btw. shadow amount and prediction confidence of the robust models
The models used to predict the object class were used from the [RobustBench](https://robustbench.github.io/#div_imagenet_Linf_heading). In order to use the robustbench models follow the instructions [here](https://github.com/RobustBench/robustbench#model-zoo-quick-tour) to install the robustbench.

The robustbench is limited to the number of test images (<=5000 images) for the ImageNet dataset. To do the model evaluation on the whole validation dataset (50.000 images) download ```imagenet_test_image_ids_all_classes.txt``` from [google-disk](), upload this file into the ```helper_files``` folder, and change the ```loaders.py``` file line 70:
```
        samples = make_custom_dataset(
            self.root, 'helper_files/imagenet_test_image_ids_all_classes.txt',
            class_to_idx)
```

The correlation analysis can be done by running the file ```robust_model_prediction_shadow_visualization.py``` in folder ```object_recognition_models```. To plot different graphs use the specified method with the specified arguments in the main path of the script ```robust_model_prediction_shadow_visualization.py```, for example:
```
val_shadow_confidence_scatter_bar_sns(n_examples, shadow_path, model_name, 
                                          dataset, threat_model, 
                                          x_test_path, y_test_path, paths_test_path)
```
and then run the script ```robust_model_prediction_shadow_visualization.py``` using ```python robust_model_prediction_shadow_visualization.py```

Here you can see the result of the correlation btw. shadow amount and prediction confidence for correctly classified samples and misclassifications on val. dataset ImageNet for the ```Liu2023Comprehensive_Swin-L``` model from robustbench.

<p align="center"><img src="imgs/Liu2023Comprehensive_Swin-L shadow prediction confidence.png" width="700">

In the same way, you can get different visualizations for different models from robustbench.
