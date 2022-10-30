import torch
import torch.nn as nn
import torch.nn.functional as F

# step1: convert image to embdding
def image2emb_naive(image, patch_size, weight):
    # image shape: B*C*H*W
    # patch shape:b*num_patch*dim
    patch = F.unfold(image, kernel_size=patch_size,stride=patch_size).transpose(-1,-2)
    patch_embedding = patch @ weight
    return patch_embedding

def image2emb_conv(image,kernel,stride):
    conv_output = F.conv2d(image,kernel,stride=stride) #B*oc*oh*ow
    bs,oc,oh,ow = conv_output.shape
    patch_embedding = conv_output.reshape(bs,oc,ow*oh).transpose(-1,-2)
    return patch_embedding

bs,ic, image_h,image_w = 1,3,8,8
patch_size = 4
model_dim = 8  #输出通道数
max_num_token = 16
num_class = 10
label = torch.randint(10,(bs,))

print(label)
print(label.shape)

patch_depth = patch_size*patch_size*ic
image = torch.randn(bs,ic,image_h,image_w)
weight = torch.randn(patch_depth,model_dim)

#分块得到emb
patch_embedding_naive = image2emb_naive(image, patch_size, weight)

#卷积得到emb
kernel = weight.transpose(0,1).reshape(-1,ic,patch_size,patch_size)
patch_embedding_conv = image2emb_conv(image,kernel,patch_size)

# print(patch_embedding_naive)
print(patch_embedding_conv.shape)

#step2: CLS token emb
cls_token_embedding = torch.randn(bs,1,model_dim,requires_grad=True)
# print(cls_token_embedding)
token_embedding = torch.cat([cls_token_embedding,patch_embedding_conv],dim=1)
# print(token_embedding)

#step3:position embedding
position_embedding_table = torch.randn(max_num_token,model_dim,requires_grad=True)
seq_len = token_embedding.shape[1]
# print(seq_len)
# print(token_embedding.shape[0])
# print(position_embedding_table[0:seq_len].shape)
position_embedding = torch.tile(position_embedding_table[0:seq_len],[token_embedding.shape[0],1,1])
# print(position_embedding_table.shape)
# print(position_embedding.shape)
token_embedding += position_embedding

#step4: pass embedding to Transformer Encoder
encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim,nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=6)
encoder_output = transformer_encoder(token_embedding)

#step5: classification
cls_token_output = encoder_output[:,0,:]
linear_layer = nn.Linear(model_dim,num_class)
logits = linear_layer(cls_token_output)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits,label)
print(loss)