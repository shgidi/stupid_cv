from train import *
from dl_open_images_data import *

if __name__ == '__main__':
    global data_type

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Data management.')
    parser.add_argument('--data_root', type=str, help='folder where image files will be stored', default="'/home/gidi/data/open images/'")
    parser.add_argument('--classes', type=str, help='', default=['Orange','Banana','Apple'],nargs='+')
    parser.add_argument('--data_type', type=str, help='', default='test')
    parser.add_argument('--cut_images', type=bool, help='cut the object from the image (one object from image)', default=False)
    
    args = parser.parse_args()
    classes = args.classes
    data_root = model_root = args.data_root
    data_root = args.data_root
    classes = args.classes
    data_type = args.data_type
    
    data_utils = Data_utils(data_root, classes, data_type)
    
    data_utils.prepare_dirs()
    
    s3, bucket = data_utils.get_s3()

    df ,class_name_count = data_utils.get_data_sets()

    # currently prepare dirs manually
    data_utils.dl_classes(df, bucket, class_name_count, args.cut_images)
    
    if args.cut_images:
        classes = [f'slice_{c}' for c in classes]
        
    train(data_root, classes)