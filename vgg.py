"""helper functions to get the content and style representations of the input image using VGG19 CNN"""
from torchvision import models
from torch import mm, t, cuda
device = 'cuda' if cuda.is_available() else 'cpu'
vgg19 = models.vgg19(pretrained = True).features.eval().to(device)

def getFeatures(image, layers = None):
	"""returns the feature representation of the input image"""
	if layers == None:
		layers = {'0':'conv1_1', 
			'5':'conv2_1', 
			'10':'conv3_1', 
			'19':'conv4_1', 
			'21':'conv4_2', 
			'28':'conv5_1'
			}
	features = {}
	x = image.clone()
	for name, layer in vgg19._modules.items():
		x = layer(x)
		if name in layers:
			features[layers[name]] = x
	return features

def grammatrix(inp):
	"""returns the gram matrix of the input"""
	b, d, h, w = inp.shape
	inp = inp.view(d, h*w)
	gram = mm(inp, inp.t())/(d*h*w)
	return gram


