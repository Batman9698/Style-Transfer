"""Helper functions for handling images"""
import os
from numpy import array
from PIL import Image
from torchvision import transforms
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

def openImage(img_path = None, shape = (500,500)):
	"""returns the specified image as a torch tensor"""
	if img_path == None:
		return('Specify a path')

	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	input_transform = transforms.Compose([transforms.Resize(shape),
						transforms.ToTensor(),
						transforms.Normalize(mean, std)
						])
	image = Image.open(img_path).convert('RGB')
	image = input_transform(image).unsqueeze(0).to(device)
	return image

def convertImage(image):
	"""converts the input image (torch tensor) into a numpy array"""
	mean = (0.485, 0.456, 0.406)
	std = (0.229, 0.224, 0.225)
	image = image.squeeze().permute(1,2,0).detach().to('cpu').numpy()
	image = image * array(std) + array(mean)
	image = image.clip(0,1)
	return image

def saveImage(image, filename = 'default_name.jpg'):
	"""converts the torch tensor into a PIL Image and saves it in the same folder"""
	image = convertImage(image)
	image = Image.fromarray(image)
	if '.jpg' in filename:
		image = image.save(os.getcwd()+'/'+filename)
	else:
		image = image.save(os.getcwd()+'/'+filename+'.jpg')
	return 'image saved'