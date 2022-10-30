import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class MultiHeadSelfAttention(nn.Module):

    def __init__(self,model_dim,num_head):
        super(MultiHeadSelfAttention,self).__init__()
        self.num_head = num_head

        self.proj_linear_layer = nn.Linear(model_dim,3*model_dim)
        self.final_linear_layer = nn.Linear(model_dim, model_dim)

    def forward(self,input,additive_mask=None):
        bs,seqlen,model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim//num_head

        proj_output = self.proj_linear_layer(input) #[bs, seqlen,model_dim*3]
        q,k,v = proj_output.chunk(3,dim=-1) # [bs, seqlen,model_dim]

        q = q.reshape(bs,seqlen,num_head,head_dim).transpose(1,2) # [bs,num_head,seqlen,head_dim]
        q = q.reshape(bs*num_head,seqlen,head_dim)

        k = k.reshape(bs,seqlen,num_head,head_dim).transpose(1,2) # [bs,num_head,seqlen,head_dim]
        k = k.reshape(bs*num_head,seqlen,head_dim)

        v = v.reshape(bs,seqlen,num_head,head_dim).transpose(1,2) # [bs,num_head,seqlen,head_dim]
        v = v.reshape(bs*num_head,seqlen,head_dim)

        if additive_mask is None :
            attn_prob = F.softmax(torch.bmm(q,k.transpose(-1,-2))/math.sqrt(head_dim),dim=-1)
        else:
            additive_mask = additive_mask.tile((num_head,1,1))
            attn_prob = F.softmax(torch.bmm(q,k.transpose(-1,-2))/math.sqrt(head_dim)+additive_mask,dim=-1)

        output = torch.bmm(attn_prob,v) # [bs*num_head,seqlen,head_dim]
        output = output.reshape(bs,num_head,seqlen,head_dim).transpose(1,2)
        output = output.reshape(bs,seqlen,model_dim)

        output = self.final_linear_layer(output)
        return attn_prob,output

def window_multi_head_self_attention(patch_embedding,mhsa,window_size=4,num_head=2):
    num_patch_in_window = window_size*window_size
    bs,num_patch,patch_depth = patch_embedding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    patch_embedding = patch_embedding.transpose(-1,-2)
    patch = patch_embedding.reshape(bs,patch_depth,image_height,image_width)
    window = F.unfold(patch,kernel_size=(window_size,window_size),stride=(window_size,window_size)).transpose(-1,-2)

    bs,num_window,patch_depth_time_num_patch_in_window = window.shape
    window = window.reshape(bs*num_window,patch_depth,num_patch_in_window).transpose(-1,-2)

    attn_prob,output = mhsa(window)

    output = output.reshape(bs,num_window,num_patch_in_window,patch_depth)
    return output

####
# shift window multi-head attention mask
def bulid_mask_for_shift_wmsa(batch_size,image_height,image_width,window_size):
    index_matrix = torch.zeros(image_height,image_width)

    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2)//window_size
            col_times = (j+window_size//2)//window_size
            index_matrix[i,j] = row_times*(image_height//window_size) + col_times + 1
    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size//2,-window_size//2),dims=(0,1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0)

    c = F.unfold(rolled_index_matrix,kernel_size=(window_size,window_size),stride=(window_size,window_size)).transpose(-1,-2)
    c = c.tile(batch_size,1,1)

    bs, num_window, num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1)
    c2 = (c1 - c1.transpose(-1,-2)) == 0
    vaild_matrix = c2.to(torch.float32)
    additive_mask = (1-vaild_matrix)*(-1e-9)

    additive_mask = additive_mask.reshape(bs*num_window, num_patch_in_window, num_patch_in_window)

    return additive_mask

# 辅助函数 window2image 将transformer block的结果转化为图片的格式
def window2image(msa_output):
    bs, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_hight = int(math.sqrt(num_window))*window_size
    image_width = image_hight

    mas_output  = msa_output.reshape(bs, int(math.sqrt(num_window)),
                                         int(math.sqrt(num_window)),
                                         window_size,
                                         window_size,
                                         patch_depth)
    mas_output = msa_output.transpose(2,3)
    image = msa_output.reshape(bs, image_hight*image_width, patch_depth)

    image = image.transpose(-1,-2).reshape(bs, patch_depth, image_hight, image_width)

    return image

#  辅助函数 shift_window 高效计算 swmsa
def shift_window(w_msa_output, window_size, shift_size, generete_mask=False):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    w_msa_output = window2image(w_msa_output)
    bs, patch_depth, image_hight, image_width = w_msa_output.shape

    rolled_w_msa_output = torch.roll(w_msa_output, shifts=(shift_size,shift_size), dims=(2,3))

    shifted_w_msa_input = rolled_w_msa_output.reshape(bs, patch_depth,
                                                       int(math.sqrt(num_window)),
                                                       window_size,
                                                       int(math.sqrt(num_window)),
                                                       window_size
                                                       )

    shifted_w_msa_input = shifted_w_msa_input.transpose(3,4)
    shifted_w_msa_input = shifted_w_msa_input.reshape(bs, patch_depth, num_window*num_patch_in_window)
    shifted_w_msa_input = shifted_w_msa_input.transpose(-1,-2)
    shifted_window = shifted_w_msa_input.reshape(bs, num_window, num_patch_in_window, patch_depth)

    if generete_mask:
        additive_mask = bulid_mask_for_shift_wmsa(bs, image_hight, image_width, window_size)
    else:
        additive_mask = None

    return shifted_window,additive_mask

def shift_window_multi_head_self_attention(w_msa_output, mhsa, window_size=4, num_head=2):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    shifted_w_msa_input, additive_mask = shift_window(w_msa_output,
                                                      window_size,
                                                      shift_size=-window_size//2,
                                                      generete_mask=True
                                                      )

    shifted_w_msa_input = shifted_w_msa_input.reshape(bs*num_window, num_patch_in_window, patch_depth)

    attn_prob, output = mhsa(shifted_w_msa_input, additive_mask=additive_mask)

    output = output.reshape(bs, num_window,num_patch_in_window,patch_depth)

    output, _ = shift_window(output, window_size, shift_size=window_size//2, generete_mask=False)

    return output

### 构建Patch Merging

class PatchMerging(nn.Module):
    def __init__(self, model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging,self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(
            model_dim*merge_size*merge_size,
            int(model_dim*merge_size*merge_size*output_depth_scale)
        )

    def forward(self,input):
        bs, num_window, num_patch_in_window, patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))

        input = window2image(input)

        merged_window = F.unfold(input, kernel_size=(self.merge_size,self.merge_size),
                                 stride=(self.merge_size,self.merge_size)
                                 ).transpose(-1,-2)

        merged_window = self.proj_layer(merged_window)

        return merged_window

### 构建 Swin TransformerBlock
class SwinTransformerBlock(nn.Module):
    def __init__(self, model_dim, window_size, num_head):
        super(SwinTransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)

        self.wsma_mlp1 = nn.Linear(model_dim, 4 * model_dim)
        self.wsma_mlp2 = nn.Linear(4 *model_dim,  model_dim)
        self.swsma_mlp1 = nn.Linear(model_dim, 4 * model_dim)
        self.swsma_mlp2 = nn.Linear(4 *model_dim,  model_dim)

        self.mhsa1 = MultiHeadSelfAttention(model_dim, num_head)
        self.mhsa2 = MultiHeadSelfAttention(model_dim, num_head)

    def forward(self, input):
        bs, num_patch, patch_depth = input.shape

        input1 = self.layer_norm1(input)
        w_msa_output = window_multi_head_self_attention(input1, self.mhsa1, window_size=4, num_head=2)
        bs, num_window, num_patch_in_window, patch_depth  =  w_msa_output.shape
        w_msa_output = input + w_msa_output.reshape(bs, num_patch, patch_depth)
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 += w_msa_output

        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(bs, num_window, num_patch_in_window, patch_depth)
        sw_msa_output = shift_window_multi_head_self_attention(input2, self.mhsa2, window_size=4, num_head=2)
        sw_msa_output = output1 + sw_msa_output.reshape(bs, num_patch, patch_depth)
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 += sw_msa_output

        output2 =output2.reshape(bs, num_window, num_patch_in_window, patch_depth)

        return output2

### 构建SwinTransformerModel
class SwinTransformerModel(nn.Module):
    def __init__(self,input_image_channel=3, patch_size=4, model_dim_C=8, num_classes=10,
                 window_size=4, num_head=2, merge_size=2):

        super(SwinTransformerModel, self).__init__()
        patch_depth = patch_size*patch_size*input_image_channel
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth,model_dim_C))
        self.block1 = SwinTransformerBlock(model_dim_C, window_size,num_head)
        self.block2 = SwinTransformerBlock(model_dim_C * 2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_C * 4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_C * 8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_C, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C * 2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C * 4, merge_size)

        self.final_layer = nn.Linear(model_dim_C*8, num_classes)

    def forward(self,image):
        patch_embedding_naive = image2emb_naive(image, self.patch_size, self.patch_embedding_weight)

        patch_embedding = patch_embedding_naive
        print(patch_embedding.shape)

        sw_msa_output = self.block1(patch_embedding)
        print("block1_output",sw_msa_output.shape)

        merged_patch1 = self.patch_merging1(sw_msa_output)
        sw_msa_output_1 = self.block2(merged_patch1)
        print("block2_output", sw_msa_output_1.shape)

        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)
        print("block3_output", sw_msa_output_2.shape)

        merged_patch3 = self.patch_merging3(sw_msa_output_2)
        sw_msa_output_3 = self.block4(merged_patch3)
        print("block4_output", sw_msa_output_3.shape)

        bs, num_window, num_patch_in_window, patch_depth = sw_msa_output_3.shape
        sw_msa_output_3 = sw_msa_output_3.reshape(bs, -1, patch_depth)

        pool_output = torch.mean(sw_msa_output_3, dim = 1)
        logits = self.final_layer(pool_output)
        print("logits",logits.shape)

        return logits

### 分类模块 代码测试
if __name__ == "__main__":
    bs, ic, image_h, image_w = 4, 3, 256, 256
    patch_size = 4
    model_dim_C = 8 #一开始的patchembedding大小
    # max_num_token = 16
    num_classes = 10
    window_size = 4
    num_head = 2
    merge_size = 2

    patch_depth = patch_size*patch_size*ic

    image = torch.randn(bs, ic, image_h, image_w)

    model = SwinTransformerModel(ic, patch_size, model_dim_C, num_classes, window_size, num_head, merge_size)
    logits = model(image)
    print(logits)