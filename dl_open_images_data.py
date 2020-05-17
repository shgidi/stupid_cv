
# file to edit: dev_nb/august_tests.ipynb
from pyforest import *
from collections import Counter

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import botocore

from PIL import Image

# for DL
from multiprocessing import  Manager
from pathos.multiprocessing import ProcessingPool as Pool
import functools
import logging
import argparse

    # fast DL with pool

n_work=10

                
# define s3 object
class Data_utils:
    def __init__(self, data_root, classes, data_type, batch_dl=1):
        self.classes = classes
        self.data_root = data_root
        self.data_type = data_type
        self.batch_dl = int(batch_dl)

    def get_s3(self):
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.bucket = "open-images-dataset"

    # some data manipulations

    def get_data_sets(self):
        classes_df = pd.read_csv(os.path.join(self.data_root,'class-descriptions-boxable.csv'),header=None)

        df = pd.read_csv(os.path.join(self.data_root,f'{self.data_type}-annotations-bbox.csv'))

        c = Counter(df.LabelName)

        class_count = pd.DataFrame.from_dict(dict(c), orient='index').reset_index()

        class_name_count=pd.merge(classes_df, class_count,left_on=0, right_on='index')[['index',1,'0_y']]
        class_name_count.columns = ['id','name','count']

        class_dict = {i[1]['name']:i[1]['count'] for i in class_name_count[['name','count']].iterrows()}

        self.id2name = {i[1].id:i[1]['name'] for i in class_name_count.iterrows()}

        return df, class_name_count

    
    def download(self, root, retry, counter, lock, path):
        i = 0
        src = '/'.join([self.data_type, path.split('/')[1]])
        dest = f"{root}/{path}"
        while i < retry:
            try:
                if not os.path.exists(dest):
                    self.s3.download_file(self.bucket, src, dest)
                else:
                    logging.debug(f"{dest} already exists.")
                with lock:
                    counter.value += 1
                    if counter.value % 100 == 0:
                        logging.warning(f"Downloaded {counter.value} images.")
                return
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    logging.warning(f"The file s3://{self.bucket}/{src} does not exist.")
                    return
                i += 1
                logging.warning(f"Sleep {i} and try again.")
                #time.sleep(i)
            
    def batch_download(self, file_paths, root, num_workers=10, retry=10):
        with Pool(num_workers) as p:
            m = Manager()
            counter = m.Value('i', 0)
            lock = m.Lock()
            download_ = functools.partial(self.download, root, 3, counter, lock)
            try:
                p.map(download_, file_paths)
            except:
                print('batch dl is cancelled due to error')
                self.batch_dl = 0

    def prepare_dirs(self):
        for c in self.classes:
            if not os.path.exists(f'{self.data_root}/{c}'):
                os.makedirs(f'{self.data_root}/{c}')

    def dl_classes(self, df, class_name_count, cut_images_flag=False):
        try:
            ids = class_name_count[class_name_count.name.isin(self.classes)].id.values
        except:
            print(f'classes {self.classes} do not exist',)
        df1 = df[df.LabelName.isin(ids)]
        imgs = df1.drop_duplicates('ImageID')

        logging.info('downloading images', Counter([self.id2name[i] for i in df1.LabelName.values]))
        for j in range(len(imgs)//n_work):
            image_files = [f'{self.id2name[i[1].LabelName]}/{i[1].ImageID}.jpg' for i in imgs[(n_work)*j:n_work*(j+1)].iterrows()]
            if self.batch_dl:
                self.batch_download(image_files, self.data_root, n_work, False)
            else:
                for file in image_files:
                    m = Manager()
                    counter = m.Value('i', 0)
                    lock = m.Lock()
                    self.download(self.data_root, 3, counter, lock, file)

        if cut_images_flag:
            print(cut_images_flag)
            self.cut_images(imgs)


    def cut_images(self, df):
        # After downloading data, generate new data folders, named sliced_X
        # TODO: allow multiple instances per image
        for record in df.iterrows():
            try:
                r = record[1]
                path = os.path.join(self.id2name[r.LabelName],f'{r.ImageID}.jpg')
                im = Image.open(os.path.join(self.data_root,path))
                cut_im = np.array(im)[int(im.size[0]*r.YMin):int(im.size[0]*r.YMax ) ,int(im.size[1]*r.XMin):int(im.size[1]*r.XMax)]
                cut_im = Image.fromarray(cut_im)
                path = os.path.join(self.id2name[r.LabelName],f'{r.ImageID}.jpg')
                dir_name = os.path.dirname(os.path.join(self.data_root, f"slice_{path}"))
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                cut_im.save(os.path.join(self.data_root,f"slice_{path}"))
            except:
                logging.warning('error')

# test
if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Data management.')
    parser.add_argument('--data_root', type=str, help='folder where image files will be stored', default="'/home/gidi/data/open images/'")
    parser.add_argument('--classes', type=str, help='', default=['Orange','Banana','Apple'],nargs='+')
    parser.add_argument('--data_type', type=str, help='', default='test')
    parser.add_argument('--batch_dl', type=str, help='', default=1)
    parser.add_argument('--cut_images', type=bool, help='cut the object from the image (one object from image)', default=False)
    args = parser.parse_args()
    
    data_root = args.data_root
    classes = args.classes
    data_type = args.data_type
    
    data_utils = Data_utils(data_root, classes, data_type, args.batch_dl)
    
    data_utils.prepare_dirs()
    
    data_utils.get_s3()

    df ,class_name_count = data_utils.get_data_sets()

    data_utils.dl_classes(df, class_name_count, args.cut_images)
    
    # TODO: add logging, add prepare dirs logic
    
    # python dl_open_images_data.py --data_root=<data_root>' --classes Apple Banana
