'''
This document outlines how to use the deep one-class classification source code.
Author : Pramuditha Perera (pramuditha.perera@rutgers.edu)


Pre-processing
--------------
1. This code is developed targeting pycaffe framework. Please make sure caffe and python 2.7 is installed.
2. Download the code into caffe/examples folder.
3. Download pre-trained models to caffe/models folder.
	For VGG16 visit : http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
	For Alexnet visit : http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
4. Download reference dataset to caffe/data. We use ImageNet validation set. It can be found at http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
5. Download target datasets to caffe/data. For novelty detection we use Caltech 256 : http://www.vision.caltech.edu/Image_Datasets/Caltech256/
   For abnormal image detection, we use Abnormal 1001 as abnormal images : http://paul.rutgers.edu/~babaks/abnormality_detection.html
   Normal image classes are taken from PASCAL VOC dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
6. Edit prototext files to reflect correct paths. Specifically, 
   In solverVGG / solverdistance files, change 'net' and 'snapshot_prefix' with correct file paths.
   In VGGjoint2 / joint2 files, change 'source'  parameter in both data and data_c layers.


Training/ Testing
-----------------


Abnormal image detection
------------------------
Two sub directories 'Abnormal_Object_Dataset' and 'Normal_Object_Dataset'
should exist in caffe/data. Each sub folder (of each class) should ne numbered started from 1.

There exists four modes of operation. To test just first class:

1. Using Alexnet features
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone Alex --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task abnormal --type feature

2. Using VGG16 features
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone VGG --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task abnormal --type feature
3. Using Alexnet DOC (ours)
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone Alex --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task abnormal

4. Using VGG16 DOC (ours)
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone VGG --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task abnormal

If all 6 classes needs to be tested replace --noneclass 6.



Novelty Detection
----------------- 

Novelty detection dataset should be stored in the  caffe/data/novelty directory. Each subfolder (of each class) should ne numbered started from 1.

There exists four modes of operation. To test just first class:

1. Using Alexnet features
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone Alex --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task novelty --type feature 

2. Using VGG16 features
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone VGG --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task novelty --type feature
3. Using Alexnet DOC (ours)
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone Alex --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task novelty

4. Using VGG16 DOC (ours)
    >>python  examples/OneClass/src/src/run.py --dataset data/ --backbone VGG --cafferoot /home/labuser/caffe/ --nclass 6 --noneclass 1 --task novelty

If 40 classes needs to be tested instead of just the first, replace --noneclass 40.


Arguments
----------
--name : Name of the network. Used to name the performance curve plot and text output containing match scores.
--type : Type of CNN : oneclass / feature. When oneclass is used classification is done using DOC. Otherwise pre-trained deep features are used.
--output : Output directory name.
--dataset : Specify the path to the training dataset. Eg: data/abnormal/
--cafferoot : Specify the path to the caffe installation. Default is : /home/labuser/caffe/
--backbone : Specify the backbone: VGG/Alex
--nclass : Number of total classes in the dataset. 256 for novelty detection and 6 for abnormal image detection.
--noneclass : Number of classes to be considered for one-class testing. We used 40 for novelty detection. 6 for abnormal image detection. 
--task : Specify oneclass task novelty/ abnormal
--niter : Number of training iterations
--visualize : True/ False specifies whether it is required to generate ROC curve plot.


output
------
A text file with one-class score values will be written to the output folder. If '--visualize' option is set to True, a ROC
curve will also be generated.

'''


from pylab import *
import caffe
import sys
import os
from random import shuffle
import writeFileNames
import classifyImage
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import argparse


def arguments():
    """Parse arguments into a parser.

    Path to the training/ testing data should be provided using '--dataset'
    argument. eg: --dataset data/abnormal/

    If the task is abnormal image detection, '--task abnormal' should be set.
    Two sub directories 'Abnormal_Object_Dataset' and 'Normal_Object_Dataset'
    should exist in data/abnormal where abnormal and normal images should be
    present.

    If the task is novelty detection (and AA), '--task novelty' should be set.
    Under the dataset directory, sub-directories should exist for each class
    numbered from 1 to N, where N is the number of classes.
	
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="oneclassVGG", help="Name of the network")
    parser.add_argument("--type", default="oneclass", help="Type of CNN : oneclass / feature ")
    parser.add_argument("--output", default='output', help="Output directory")
    parser.add_argument("--dataset", default="data/abnormal/",
                        help="Specify the path to the training dataset")
    parser.add_argument("--cafferoot", default="/home/labuser/caffe/",
                        help="Specify the path to the caffe instalation")
    parser.add_argument("--backbone", default="VGG",
                        help="Specify the backbone: VGG/Alex")
    parser.add_argument("--nclass", default=256,
                        help="Number of total classes in the dataset")
    parser.add_argument("--noneclass", default=40,
                        help="Number of classes to be considered for one-class testing")
    parser.add_argument("--task", default='abnormal',
                        help="Specify oneclass task novelty/ abnormal")
    parser.add_argument("--niter", default=700,
                        help="Training iterations")
    parser.add_argument("--visualize", default=True,
                        help="Save ROC curves")

    return(parser)


parser = arguments()
physical_dir = os.path.dirname(os.path.realpath(__file__))
args = parser.parse_args()
caffe_root = args.cafferoot
sys.path.insert(0, caffe_root + 'python')
caffe.set_device(0)
caffe.set_mode_gpu()
subpath = args.dataset
path = caffe_root+subpath
niter = int(args.niter)
users = range(1,int(args.nclass)+1)
if not os.path.isdir(physical_dir+'/'+args.output):
	os.mkdir(physical_dir+'/'+args.output)


for user_no in range(1,int(args.noneclass)+1):
	os.chdir(caffe_root)	
	print("Writing files done for user"+ str(users[user_no-1]) +"...")
	if args.task == "novelty":
		writeFileNames.write(user_no, users, path,subpath, physical_dir+"/")
	elif args.task == "abnormal":
		writeFileNames.writeAbnormal(user_no, users, path,subpath, physical_dir+"/")
	solver = None  
	if args.type == "oneclass":
		if args.backbone == "VGG":
			solver = caffe.SGDSolver(physical_dir+'/solverVGG.prototxt')
			solver.net.copy_from('models/VGG_ILSVRC_16_layers.caffemodel')
		elif args.backbone == "Alex":
			solver = caffe.SGDSolver(physical_dir+'/solverdistance.prototxt')
			solver.net.copy_from('models/bvlc_alexnet.caffemodel')
		
		for it in range(niter):
    			solver.step(1)  
  			if args.backbone == "VGG":
    				solver.test_nets[0].forward(start='conv1_1')
  			elif args.backbone == "Alex":
    				solver.test_nets[0].forward(start='conv1')
	
	print("Classifying files...")
	os.chdir(physical_dir)
	if args.backbone == "VGG":
		model_def = 'deploy.prototxt'
		if args.type == "oneclass":
			model_weights = 'VGG_JOINT_layers_iter_'+str(niter)+'.caffemodel'
		else:
			model_weights = caffe_root+ 'models/VGG_ILSVRC_16_layers.caffemodel'
		fpr,tpr,roc_auc = classifyImage.getResults(model_def,model_weights,args.name
							  +args.name+'_'+str(users[user_no-1])+args.backbone+'_'+args.type+".txt",20,'VGG',caffe_root )

	elif args.backbone == "Alex":
		model_def = 'deploy_alex.prototxt'
		if args.type == "oneclass":
			model_weights = 'Alex_JOINT_layers_iter_'+str(niter)+'.caffemodel'
		else:
			model_weights = caffe_root + 'models/bvlc_alexnet.caffemodel'
		fpr,tpr,roc_auc = classifyImage.getResults(model_def,model_weights,physical_dir+'/'+args.output+'/'
						       +args.name+'_'+str(users[user_no-1])+args.backbone+'_'+args.type+".txt",40,'Alex',caffe_root )
	print('Area under the curve: ' + str(roc_auc))
	if args.visualize:
		fig = plt.figure()
		plt.plot(fpr, tpr,lw=2, label='ROC curce ' + str(roc_auc))
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic')
		plt.legend(loc="lower right")
		plt.savefig(physical_dir+'/'+args.output+'/'+args.name+'_'+str(users[user_no-1])+args.backbone+'_'+args.type+'.png') 
		plt.close("all")












