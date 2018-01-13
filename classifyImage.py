def getFeature (imagepath,transformer, net):
	image = caffe.io.load_image(imagepath)

	net.blobs['data'].data[...] = image
	output = net.forward()
	feature= output['fc7'][0]  # the output probability vector for the first image in the batch
	return feature


def getFeaturesFromMatrix (matrix,transformer, net):
	net.blobs['data'].data[...] = matrix
	output = net.forward()
	feature= output['fc7']
	return feature


def getFeaturesFromMatrixBi (matrix,transformer, net):
	net.blobs['data'].data[...] = matrix
	output = net.forward()
	feature= output['fc8']
	return feature[0]


def getNet (model_def ,model_weights, csize, caffepath, bs):
	import caffe
	import numpy as np
	net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  	# contains the trained weights
                caffe.TEST)     	# use test mode (e.g., don't perform dropout)
	net.blobs['data'].reshape(bs,        # batch size
                          3,         	     # 3-channel (BGR) images
                          csize, csize)  	

	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	
	mu = np.load(caffepath + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
	mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)  
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
	return net, transformer


def getResults(model_def,model_weights,outfile,bsize,nw,caffe_root):
	import numpy as np
	from random import shuffle
	import os
	from sklearn.metrics import roc_curve, auc
	import sys
	import caffe
	sys.path.insert(0, caffe_root + 'python')
	caffepath = caffe_root
	caffe.set_device(0)
	caffe.set_mode_gpu()
	if nw == 'VGG':
		loadsz = 20
		csize = 224
	else:
		loadsz = 40
		csize = 227

	#Get validation data set
	f = open("signature.txt")
	signature_files=[];
	for l in f.readlines():
	   currFileNames = l.strip().split(" " );  
 	   signature_files.append(currFileNames[0])
	f.close()

	net, transformer = getNet (model_def ,model_weights, csize, caffepath, loadsz);	
	max_sig_size =bsize;
	images = np.empty((max_sig_size,3,csize,csize))
	count = 0;
	probeimages = np.zeros((max_sig_size,3,csize,csize))
	signature = np.zeros((len(signature_files),4096)) 
	nblocks = 0;
	for l in range(len(signature_files)):  
 	  image = caffe.io.load_image(caffepath+signature_files[l])	
	  transformed_image = transformer.preprocess('data', image)
 	  images[count,:,:,:] = transformed_image;
	  count = count+1;
	  
		
	  if count==max_sig_size:
		count = 0;
		probefeatures = getFeaturesFromMatrix (images,transformer, net)		
		signature[nblocks*max_sig_size:(nblocks+1)*max_sig_size,:]=getFeaturesFromMatrix (images,transformer, net)
		nblocks+=1
	#Get test data and compare with validation data
	f = open("test.txt")
	testfiles=[];
	labels=[];
	for l in f.readlines():
  	 currFileNames = l.strip().split(" " );  
  	 testfiles.append(currFileNames[0])
  	 labels.append(int(currFileNames[1])) # matched
	f.close()
	matched=[];

	#Testing
	print("TESTING..")
	max_test_size =bsize;
	count=0;
	probeimages = np.zeros((max_test_size,3,csize,csize))
	for n in testfiles:
		image = caffe.io.load_image(caffepath+n)	
   		transformed_image = transformer.preprocess('data', image)
   		probeimages[count,:,:,:] = transformed_image;
		count = count+1
		if count==max_test_size:
			#batch is complete
			count = 0;
			probefeatures = getFeaturesFromMatrix (probeimages,transformer, net)		
			probeimages = np.zeros((max_test_size,3,csize,csize))
			for c in range(np.shape(probefeatures)[0]):
				vec = probefeatures[c,:];
				#get miniumum cosine distance
				distance=[];
				for p in range(len(signature_files)):	
					sig = signature[p,:]			
					distance.append(np.sqrt(float(np.dot(vec-signature[p],vec-signature[p]))));
				matched.append((-1)*min(distance))
	labels= labels[0:len(matched)]
	text_file = open(str(outfile), "w")
	for x in range(len(matched)):
		text_file.write("%s %s\n" % (str(matched[x]), str(labels[x])))
	text_file.close()
	fpr, tpr, _ = roc_curve(labels, matched, 0)
	roc_auc = auc(fpr, tpr)
	return(fpr,tpr,roc_auc)




def getResultsPlus(model_def,model_weights,outfile,bsize,nw,caffe_root):
	import numpy as np
	from random import shuffle
	import os
	import sys
	sys.path.insert(0, caffe_root + 'python')
	import caffe
	caffepath = caffe_root
	caffe.set_device(0)
	caffe.set_mode_gpu()
	if nw == 'VGG':
		loadsz = 20
		csize = 224
	else:
		loadsz = 40
		csize = 227



	net, transformer = getNet (model_def ,model_weights, csize, caffepath, loadsz);	
	max_sig_size =bsize;
	images = np.empty((max_sig_size,3,csize,csize))
	nblocks = 0;

	f = open("test.txt")
	testfiles=[];
	labels=[];
	for l in f.readlines():
  	 currFileNames = l.strip().split(" " );  
  	 testfiles.append(currFileNames[0])
  	 labels.append(int(currFileNames[1])) # matched
	f.close()
	matched=[];


	print("TESTING..")
	max_test_size =bsize;
	count=0;
	probeimages = np.zeros((max_test_size,3,csize,csize))
	for n in testfiles:
		image = caffe.io.load_image(caffepath+n)	
   		transformed_image = transformer.preprocess('data', image)
   		probeimages[count,:,:,:] = transformed_image;
		count = count+1
		if count==max_test_size:
			#batch is complete
			count = 0;
			net, transformer = getNet (model_def ,model_weights,  csize, caffepath, loadsz);
			probefeatures = getFeaturesFromMatrixBi (probeimages,transformer, net)		
			probeimages = np.zeros((max_test_size,3,csize,csize))
			matched= matched+probeimages
	


	labels= labels[0:len(matched)]

	#Lets write results to a text file 
	text_file = open(str(outfile), "w")
	for x in range(len(matched)):
		text_file.write("%s %s\n" % (str(matched[x]), str(labels[x])))
	text_file.close()
	fpr, tpr, _ = roc_curve(labels, matched, 0)
	roc_auc = auc(fpr, tpr)
	return(fpr,tpr,roc_auc)


