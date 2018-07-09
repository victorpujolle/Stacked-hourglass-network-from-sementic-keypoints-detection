"""
training and validation on the same image folder
"""






from hourglass_dann_v10 import HourglassModel
from time import time, clock
import numpy as np
import tensorflow as tf
import scipy.io
import cv2
from predictclass2 import PredictProcessor
from yolo_net import YOLONet
from datagen import DataGenerator
import config as cfg
from filters import VideoFilters
import os
import configparser

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)

	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


class Inference():
    """ Inference Class
    Use this file to make your prediction
    Easy to Use
    Images used for inference should be RGB images (int values in [0,255])
    Methods:
        webcamSingle : Single Person Pose Estimation on Webcam Stream
        webcamMultiple : Multiple Person Pose Estimation on Webcam Stream
        webcamPCA : Single Person Pose Estimation with reconstruction error (PCA)
        webcamYOLO : Object Detector
        predictHM : Returns Heat Map for an input RGB Image
        predictJoints : Returns joint's location (for a 256x256 image)
        pltSkeleton : Plot skeleton on image
        runVideoFilter : SURPRISE !!!
    """

    def __init__(self, config_file='config.cfg', model='hg_refined_tiny_200', yoloModel='YOLO_small.ckpt'):
        """ Initilize the Predictor
        Args:
            config_file 	 	: *.cfg file with model's parameters
            model 	 	 	 	: *.index file's name. (weights to load)
            yoloModel 	 	: *.ckpt file (YOLO weights to load)
        """
        t = time()
        params = process_config(config_file)
        self.predict = PredictProcessor(params)
        self.predict.color_palette()
        self.predict.LINKS_JOINTS()
        self.predict.model_init()
        self.predict.load_model(load=model)
        self.predict.yolo_init()
        self.predict.restore_yolo(load=yoloModel)
        self.predict._create_prediction_tensor()
        self.filter = VideoFilters()
        print('Done: ', time() - t, ' sec.')

    # ----------------------- Heat Map Prediction ------------------------------

    def predictHM(self, img):
        """ Return Sigmoid Prediction Heat Map
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
        """
        # 		return self.predict.pred(self, img / 255, debug = False, sess = None)
        return self.predict.pred(img / 255, debug=False, sess=None)

    # ------------------------- Joint Prediction -------------------------------

    def predictJoints(self, img, mode='cpu', thresh=0.2):
        """ Return Joint Location
        /!\ Location with respect to 256x256 image
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
            mode : 'cpu' / 'gpu' Select a mode to compute joints' location
            thresh : Joint Threshold
        """
        SIZE = False
        if len(img.shape) == 3:
            batch = np.expand_dims(img, axis=0)
            SIZE = True
        elif len(img.shape) == 4:
            batch = np.copy(img)
            SIZE = True
        if SIZE:
            if mode == 'cpu':
                return self.predict.joints_pred_numpy(batch / 255, coord='img', thresh=thresh, sess=None)
            elif mode == 'gpu':
                return self.predict.joints_pred(batch / 255, coord='img', debug=False, sess=None)
            else:
                print("Error : Mode should be 'cpu'/'gpu'")
        else:
            print('Error : Input is not a RGB image nor a batch of RGB images')

    # ----------------------------- Plot Skeleton ------------------------------

    def pltSkeleton(self, img, thresh, pltJ, pltL):
        """ Return an image with plotted joints and limbs
        Args:
            img : Input Image -shape=(256x256x3) -value= uint8 (in [0, 255])
            thresh: Joint Threshold
            pltJ: (bool) True to plot joints
            pltL: (bool) True to plot limbs
        """
        return self.predict.pltSkeleton(img, thresh=thresh, pltJ=pltJ, pltL=pltL, tocopy=True, norm=True)

    # -------------------------- Process Stream --------------------------------

    def centerStream(self, img):
        img = cv2.flip(img, 1)
        img[:,
        self.predict.cam_res[1] // 2 - self.predict.cam_res[0] // 2:self.predict.cam_res[1] // 2 + self.predict.cam_res[
            0] // 2]
        img_hg = cv2.resize(img, (256, 256))
        img_res = cv2.resize(img, (800, 800))
        img_hg = cv2.cvtColor(img_hg, cv2.COLOR_BGR2RGB)
        return img_res, img_hg

    def plotLimbs(self, img_res, j):
        """
        """
        for i in range(len(self.predict.links)):
            l = self.predict.links[i]['link']
            good_link = True
            for p in l:
                if np.array_equal(j[p], [-1, -1]):
                    good_link = False
            if good_link:
                pos = self.predict.givePixel(l, j)
                cv2.line(img_res, tuple(pos[0])[::-1], tuple(pos[1])[::-1], self.predict.links[i]['color'][::-1],
                         thickness=5)




if __name__ == '__main__':
    for dr in np.linspace(0.1,0.3,20):

        network_name = '../trained_networks/hg_test_32_dr_' + str(round(dr,2))

        try :
            print('--Parsing Config File')
            params = process_config('config.cfg')

            print('--Creating Dataset')
            dataset = DataGenerator(params['joint_list'], params['img_directory'], params['training_txt_file'],
                                    remove_joints=params['remove_joints'])
            dataset._create_train_table()
            dataset._randomize()
            dataset._create_sets()
            # model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'], training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'], joints= params['joint_list'], modif=False)

            model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'],
                                   nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'],
                                   drop_rate=params['dropout_rate'], lear_rate=round(dr,2),
                                   decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset,
                                   training=True, logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'],
                                   name=network_name, joints=params['joint_list'])

            model.generate_model()
            model.training_init(nEpochs=50, epochSize=params['epoch_size'], saveStep=params['saver_step'],
                                dataset=None)



        except:
            pass

