import logging
import argparse
import pandas as pd
from collections import Counter
if __name__=="__main__":
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Data management.')
    parser.add_argument('--data_root', type=str, help='folder where image files will be stored', default="'/home/gidi/data/open images/'")
    parser.add_argument('--data_type', type=str, help='', default='test')
    args = parser.parse_args()
    
    data_root = args.data_root
    data_type = args.data_type

    classes = pd.read_csv(f'{data_root}/class-descriptions-boxable.csv',header=None)
    df = pd.read_csv(f'{data_root}/{data_type}-annotations-bbox.csv')
    c = Counter(df.LabelName)
    class_count = pd.DataFrame.from_dict(dict(c),orient='index').reset_index()

    class_name_count=pd.merge(classes,class_count,left_on=0,right_on='index')[['index',1,'0_y']]
    class_name_count.columns = ['id','name','count']
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(class_name_count)
