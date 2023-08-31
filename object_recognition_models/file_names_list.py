import os

def imagenet_test_image_ids_class_txt(folder = './data/val/'):
    subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    for subf in subfolders:
        subf_path = folder + subf
        for path in os.scandir(subf_path):
            if path.is_file():
                print(subf+'/'+path.name)
                with open('./data/val/'+subf+'/'+subf+'.txt', 'a') as f:
                    f.write(subf+'/'+path.name+'\r\n')

def imagenet_test_image_ids_all_classes_txt(folder = './data/val/'):
    subfolders = [ f.name for f in os.scandir(folder) if f.is_dir() ]
    list_test_names_path = folder + 'imagenet_test_image_ids_all_classes.txt'
    with open(list_test_names_path, 'a') as f:
        for subf in subfolders:
            class_name = subf
            list_test_names_class_path = folder + class_name + '/' + class_name +'.txt'

            print(class_name, len(open(list_test_names_class_path, 'r').readlines()))
            for line in open(list_test_names_class_path, 'r'):
                f.write(line)


if __name__ == '__main__':
    # create the txt-file for robustbench test data loading
    # upload the file into the ../robustbench/helper_files and change the line 70 in loaders.py for the path to the new file
    imagenet_test_image_ids_all_classes_txt(folder = './data/val/')