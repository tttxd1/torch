import torch
import torch.nn as nn
import imageio
from Unet import UNet

model = UNet(num_classes=7)
model.load_state_dict(torch.load( r"E:\yqj\try\code\torch\Train\save_model\UNet\256_max\350-0.01595.pth",map_location = torch.device('cpu')))

image = r"E:\yqj\try\code\torch\Train\Data\coastline\train\image\1_1_1.tif"
image = imageio.imread(image)
image = torch.from_numpy(image)
image = image.permute(2,0,1)
image = torch.unsqueeze(image,0)

# image = image.reshape([1,4,512,512])

result = model(image)

savepath = r"E:\yqj\try\code\torch\Train\Data\coastline\resulrt\1.tif"

output = torch.squeeze(result, 0)
output = torch.argmax(output, 0).numpy().astype('uint8')
imageio.imwrite(savepath,output)