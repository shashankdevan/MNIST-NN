import numpy, idx2numpy

def load_dataset(IDX_IMG_FILE, IDX_LBL_FILE):
	imgs = idx2numpy.convert_from_file(IDX_IMG_FILE)
	lbls = idx2numpy.convert_from_file(IDX_LBL_FILE)
	lbls = lbls.tolist()
	train = []
	for i in range(len(imgs)):
		vector = []
		for row in imgs[i]:
			vector = vector + row.tolist()
		img = []
		img.append(vector)
		img.append(lbls[i])
		train.append(img)
	print len(train)
	for i in range(10):
		print train[i]

load_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
