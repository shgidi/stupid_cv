import os
import glob
from fastai import *
from fastai.vision import *   # Quick access to computer vision functionality
from fastai.callbacks import *
import time
from mlflow_tracker import *
from collections import Counter
import logging
import argparse

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
training_params = config['training_params']


def get_dataframe(data_root, classes):
    # return a dataframe by files in train folder
    files = []
    for c in classes:
        files+=glob.glob(data_root+f'/{c}/*.*') 
    print(f'caught {len(files)} files')
    cats = [re.split('/',file)[-2] for file in files]
    
    parsed_files = ['/'.join(file.split('/')[-2:]) for file in files]
    df = pd.DataFrame({'name':parsed_files,'label':cats})
    print(Counter(df.label))
    return df

def train(data_root, classes):

    img_sz = training_params['img_sz']
    bs = training_params['bs']
    valid_pct = training_params['valid_pct']
    df = get_dataframe(data_root, classes)
    
    tfms = get_transforms(xtra_tfms=zoom_crop(scale=(0.75,2), do_rand=True))
    data = ImageDataBunch.from_df(data_root,df,bs=bs,valid_pct=.2,size=img_sz,ds_tfms=tfms)
    # max bs is 4 for these setings
    data.normalize(imagenet_stats)

    MLLogger = partial(MLFlowTracker, nb_path = os.path.abspath(__file__),
                       exp_name="open_images",params = training_params)

    model_name = 'resnet34'
    models_dict = {'resnet18':models.resnet18,'resnet34':models.resnet34}

    # define learner
    models_dir_time = os.path.join(data_root,'models',time.strftime("%d-%m-%y-%H-%M-%S"))
    
    model = models_dict[model_name]

    precision = Precision()
    recall = Recall()
    fbeta = FBeta(beta=0.5)
    learn = cnn_learner(data, model, metrics=[accuracy,precision, recall,fbeta], model_dir=Path(models_dir_time)
                      , callback_fns=[CSVLogger]) #MLLogger

    epochs = training_params['epochs']
    lr = training_params['lr']
    mixup = 0.2
    learn = learn.mixup(alpha=mixup)
    augs = {'mixup':mixup}
    
    #logging.info(f'training {model_name} on {len(data)} images')
    learn.fit(epochs,lr, callbacks=[SaveModelCallback(learn, every='improvement', monitor='f_beta')])
    logging.info(f'bestmodel.pth saved at {models_dir_time}')

# test
if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description='Data management.')
    parser.add_argument('--data_root', type=str, help='folder where image files will be stored', default="'/home/gidi/data/open images/'")
    parser.add_argument('--classes', type=str, help='', default=['Orange','Banana','Apple'],nargs='+')
    args = parser.parse_args()
    classes = args.classes
    data_root = model_root = args.data_root
    
    train(data_root, classes)