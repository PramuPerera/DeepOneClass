from pylab import *
import matplotlib.pyplot as plt
import caffe
import sys
import os
import glob
from random import shuffle


def writeAbnormal (user_no, users, path, subpath,curr_path):

	print("Writing files for user"+str(users[user_no-1]))
	dirs = os.listdir(path+'Normal_Object_Dataset/'+str(users[user_no-1]));
	shuffle(dirs)
	intruder_classes = set(users).difference(set([user_no])); 
	text_file = open(curr_path+"train.txt", "w")
	for x in range(0, int(floor(size(dirs)*0.5))):
		text_file.write("%s%s/%s 0\n" % (subpath+'Normal_Object_Dataset/',str(users[user_no-1]),dirs[x]))
	text_file.close()


	text_file = open(curr_path+"val.txt", "w")
	for x in range(int(floor(size(dirs)*0.4)), int(floor(size(dirs)*0.5))):
		text_file.write("%s%s/%s 0\n" % (subpath+'Normal_Object_Dataset/',str(users[user_no-1]),dirs[x]))
	text_file.close()

	text_file = open(curr_path+"signature.txt", "w")
	for x in range(int(floor(size(dirs)*0.5)-40), int(floor(size(dirs)*0.5))):
		text_file.write("%s%s/%s 0\n" % (subpath+'Normal_Object_Dataset/',str(users[user_no-1]),dirs[x]))
	text_file.close()

	text_file = open(curr_path+"test.txt", "w")
	for x in range(int(floor(size(dirs)*0.5)), int(size(dirs))):
		text_file.write("%s%s/%s 0\n" % (subpath+'Normal_Object_Dataset/',str(users[user_no-1]),dirs[x]))

	items_per_intruder_class = int(floor(size(dirs)*0.5));
	
	dirs = os.listdir(path+'Abnormal_Object_Dataset/'+str(users[user_no-1]));
	shuffle(dirs)
	for x in range(0,len(dirs)):
		text_file.write("%s%s/%s 1\n" % (subpath+'Abnormal_Object_Dataset/',str(users[user_no-1]),dirs[x]))
	
	text_file.close()


def write (user_no, users, path, subpath,curr_path):

	print("Writing files for user"+str(user_no))
	dirs = os.listdir(path+'/'+str(user_no));
	shuffle(dirs)
	intruder_classes = set(users).difference(set([user_no])); 
	text_file = open(curr_path+"train.txt", "w")
	for x in range(0, int(floor(size(dirs)*0.5))):
		text_file.write("%s%s/%s 0\n" % (subpath+'/',str(user_no),dirs[x]))
	text_file.close()


	text_file = open(curr_path+"val.txt", "w")
	for x in range(int(floor(size(dirs)*0.4)), int(floor(size(dirs)*0.5))):
		text_file.write("%s%s/%s 0\n" % (subpath+'/',str(user_no),dirs[x]))
	text_file.close()

	text_file = open(curr_path+"signature.txt", "w")
	for x in range(int(floor(size(dirs)*0.5)-40), int(floor(size(dirs)*0.5))):
		text_file.write("%s%s/%s 0\n" % (subpath+'/',str(user_no),dirs[x]))
	text_file.close()

	text_file = open(curr_path+"test.txt", "w")
	for x in range(int(floor(size(dirs)*0.5)), int(floor(size(dirs)))):
		text_file.write("%s%s/%s 0\n" % (subpath+'/',str(user_no),dirs[x]))

	items_per_intruder_class = int(floor(0.5*size(dirs)/len(intruder_classes)));
	print(users)

	for y in intruder_classes :
		dirs = os.listdir(path+'/'+str(y));
		shuffle(dirs)
		for x in range(0, items_per_intruder_class):
			text_file.write("%s%s/%s 1\n" % (subpath+'/',str(y),dirs[x]))
	

	text_file.close()

