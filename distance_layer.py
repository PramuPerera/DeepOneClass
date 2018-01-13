import caffe
import numpy as np
import math


class DistanceLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        assert len(bottom) == 1,            'requires two layer.bottom'
        assert bottom[0].data.ndim == 2,    'requires blobs of one dimension data: FC feature'
        assert len(top) == 1,               'requires a single layer.top'

    def reshape(self, bottom, top):

        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
	n = np.shape(bottom[0].data)[0]; #Number of rows --> batch size
	m = np.shape(bottom[0].data)[1];
	loss = np.zeros(n, dtype=np.float32);
	
	for i in range(0,n):
		probe = bottom[0].data[i,:];
		others = np.delete(bottom[0].data,i,axis=0);
		# Get the mean of other vectors
		mean_vec = np.sum(others,axis=0)/float(n-1);
		for j in range(0,np.shape(bottom[0].data)[1]):	
				loss[i] = loss[i]+ math.pow((bottom[0].data[i][j]-mean_vec[j])/float(n),2);
        top[0].data[...] = np.sum(loss);
	self.diff = loss;

    def backward(self, top, propagate_down, bottom):
	m = np.shape(bottom[0].data)[1];
	n = np.shape(bottom[0].data)[0]; 
	deri = np.zeros_like(bottom[0].data, dtype=np.float32);
	for i in range(0,n):


		probe = bottom[0].data[i,:];
		others = np.delete(bottom[0].data,i,axis=0);
		mean_vec = np.sum(others,axis=0)/float(n-1);
		cum_der = 0


		for j in range(0,np.shape(bottom[0].data)[1]):	

				temp = m/(2*n)*2*(bottom[0].data[i][j]-mean_vec[j])/((float(n*m))) 

				deri[i][j] = (float(n)/(n-1))*temp
				cum_der = cum_der + temp


		deri[i][:] = deri[i][:]-cum_der/float(n-1)
				
	bottom[0].diff[...] = deri[...];
				

