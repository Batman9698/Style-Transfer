import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np 
import os
import matplotlib.pyplot as plt 
from PIL import Image

def openImage(path, shape = 500):
	input_transform = transforms.Compose([transforms.Resize((shape,shape)),
											transforms.ToTensor(),
											transforms.Normalize((0.485, 0.456, 0.406),
																	(0.229, 0.224, 0.225))
											])
	image = Image.open(path).convert('RGB')
	image = input_transform(image).unsqueeze(0)
	return image 
def convert(image):
	image = image.squeeze().permute(1,2,0).detach().to("cpu").numpy()
	image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
	return image 
def getFeatures(image, model):
	x = image 
	layers = ['0','3','6','8','11','13','16','18']
	features = []
	for name, layer in model._modules.items():
		x = layer(x)
		if name in layers:
			features.append(x)
	return features 
def getGramMatrix(inp):
	b,d,h,w = inp.size()
	inp = inp.view(d, h*w)
	gram = torch.mm(inp,inp.t())
	return gram 

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:",device)

vgg = models.vgg11(pretrained = True).features
for parameter in vgg.parameters():
	parameter.requires_grad_(False)
vgg = vgg.to(device)

content_image = openImage(os.getcwd()+'/me1.jpg').to(device)
style_image = openImage(os.getcwd()+'/paint.jpeg').to(device)
image_features = getFeatures(content_image, vgg_features)
image_style = [getGramMatrix(feature) for feature in getFeatures(style_image, vgg_features)]
target_image = content_image.clone().requires_grad_(True).to(device)

optimizer = torch.optim.Adam([target_image], lr = 0.001)

epochs = 1
alpha = 10
beta = 5
style_weights = [1, 0.8, 0.7, 0.6, 0.5, 0.3, 0.2, 0.1]
loss = []

for epoch in range(epochs):
    
    target_features = getFeatures(target_image, vgg_features)
    content_loss = torch.mean((target_features[4]-image_features[4])**2)
    
    style_loss = 0.0
    target_style = [getGramMatrix(target_feature) for target_feature in target_features]
    for i in range(8):
        _,d,h,w = target_features[i].shape
        style_loss += style_weights[i]*(torch.mean((target_style[i] - image_style[i])**2)/(d*h*w))
    
    total_loss = alpha*content_loss + beta*style_loss
    loss.append(total_loss.item())
    print(total_loss.item())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

plt.imshow(convert(target_image))
plt.show()