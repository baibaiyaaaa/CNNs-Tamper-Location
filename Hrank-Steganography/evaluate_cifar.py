import math
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from copy import deepcopy
import hashlib
from collections import OrderedDict
from models.cifar10.vgg import vgg_16_bn
from data import cifar10
from stego import FloatBinary, str_to_bits, bits_to_str, dummy_data_generator

parser = argparse.ArgumentParser("Cifar-10 training")

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='path to dataset')

parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    help='architecture')

parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='batch size')

parser.add_argument(
    '--rank_conv_prefix',
    type=str,
    default='./rank_conv/vgg_16_bn',
    help='rank conv file folder')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./pre_trained/vgg_16_bn.pt',
    help='pretrain model path')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASSES = 10
print_freq = (256*50)//args.batch_size

# How many bits (LSB) to use from the fraction (mantissa) of the float values
BITS_TO_USE = 1
assert BITS_TO_USE <= 23, "Can't be bigger then 23 bits"

# load training data
val_loader = cifar10.load_data(args)

# load model
model = eval(args.arch)(compress_rate=[0.] * 100).cuda()
#print(model)
ckpt = torch.load(args.pretrain_dir, map_location='cuda:0')
model.load_state_dict(ckpt['state_dict'])
state_dict = model.state_dict()
#print(state_dict)

# Data storage capacity of the layers
layers_storage_capacity_mb: Dict[str, int] = {}
layernum = 1
prefix = args.rank_conv_prefix + '/rank_conv'
subfix = ".npy"
array = []
selected_layers_weights = []
original_weights_dict = {}
for name, module in model.named_modules():
    name = name.replace('module.', '')
    if isinstance(module, nn.Conv2d):
        if layernum <= 12:
            oriweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            #print(oriweight.shape)
            #print(orifilter_num)
            #print(oriweight[37, :, :, :])
            #print("转换后的张量对象类型: ", oriweight[0, :, :, :].dtype)
            #print("转换后的张量对象类型: ", oriweight[0, :, :, :].type())

# Data storage capacity of the layers
            nb_params = np.prod(oriweight.shape)  #图层的权重存储在layer.get_weights()[0]，偏差存储在layer.get_weights()[1]
            capacity_in_bytes = np.floor((nb_params * BITS_TO_USE) / 8).astype(int)
            layers_storage_capacity_mb[name] = capacity_in_bytes / float(1<<20)
            #layers_storage_capacity_mb[name] = capacity_in_bytes

#sort rank
            cov_id = layernum
            rank = np.load(prefix + str(cov_id) + subfix)
            select_index = np.argsort(rank)[:3]
            select_index.sort()
            #print(select_index)
            array.append(select_index)

# All the Conv2D layers
            select_filter_list = []
            for i in select_index:
                select_filter_list.append(oriweight[i, :, :, :])
                v = oriweight[i, :, :, :].cpu().numpy().ravel()
                selected_layers_weights.extend(v)

# Store the original weights
# This dict holds the original weights for the selected layers
            origin_filter_dic = OrderedDict(zip(select_index, select_filter_list))

        #original_weights_dict[list(layers_storage_capacity_mb.keys())[-1]] = origin_filter_dic
        if name not in layers_storage_capacity_mb.keys():
            break
        #print(name)
        original_weights_dict[name] = origin_filter_dic
        layernum += 1

#print(original_weights_dict)
#print(array)
#print(layers_storage_capacity_mb)
layer_names = list(layers_storage_capacity_mb.keys())
selected_layers_weights = np.array(selected_layers_weights)
#print(len(selected_layers_weights))
nb_values = len(selected_layers_weights)
overall_storage_capacity_bytes = nb_values * BITS_TO_USE / 8
overall_storage_capacity_mb = overall_storage_capacity_bytes // float(1<<20)


print(f"""Stats for {layer_names}
---
(Maximum) Storage capacity is {overall_storage_capacity_mb} MB for the {len(layer_names)} layers with the {BITS_TO_USE} bits modification
""")


# Hide a secret in the layer

# The secret
f = open("meta.txt",encoding = "utf-8")
#print(f.read())
secret_to_hide = f.read()
f.close()
#print(secret_to_hide)
secret_bits = str_to_bits(secret_to_hide)
#secret = bits_to_str(secret_bits)
#print(secret)
nb_vals_needed = math.ceil(len(secret_bits) / BITS_TO_USE)
print(f"We need {nb_vals_needed} float values to store the info\nOverall number of values we could use: {nb_values}")


# Hide it

# Create the modified ones
# This dict will hold the modified (secret hidden) weights for the layers
modified_weights_dict = deepcopy(original_weights_dict)

for n in modified_weights_dict:
    weights = []
    for num in modified_weights_dict[n]:
        weights.append(modified_weights_dict[n][num])
    weights = torch.stack(weights)
    modified_weights_dict[n] = weights
#print(modified_weights_dict)
# Index of the last value used in a layer - this is needed because we don't necessary need
# the same number of params for hiding some bits then all the layer parameters
last_index_used_in_layer_dict: dict = {}

# Variable which tracks the number of values changed so far (used to index the secret message bits)
i = 0

for n in modified_weights_dict:
    #print(n)
    #print('-'*100)
    # Check if we need more values to use to hide the secret, if not then we are done with modifying the layer's weights
    if i >= nb_vals_needed:
        break

    w = modified_weights_dict[n]
    w_shape = w.shape
    #print(w_shape)
    #print('-'*100)
    w = w.ravel()

    nb_params_in_layer: int = np.prod(w.shape)
    print(nb_params_in_layer)


    for j in range(nb_params_in_layer):
        # Chunk of data from the secret to hide
        _from_index = i * BITS_TO_USE
        _to_index = _from_index + BITS_TO_USE
        bits_to_hide = secret_bits[_from_index:_to_index]
        bits_to_hide = list(map(bool, bits_to_hide))

        # Modify the defined bits of the float value fraction
        x = FloatBinary(w[j])
        fraction_modified = list(x.fraction)
        if len(bits_to_hide) > 0:
            fraction_modified[-BITS_TO_USE:] = bits_to_hide

        x_modified = x.modify_clone(fraction=tuple(fraction_modified))
        w[j] = x_modified.v

        i += 1

        # Check if we need more values to use to hide the secret in the current layer, if not then we are done
        if i >= nb_vals_needed:
            break

    last_index_used_in_layer_dict[n] = j
    w = w.reshape(w_shape)
    modified_weights_dict[n] = w

    print(f"Layer {n} is processed, last index modified: {j}")

# Load the modified (secret is hidden) weights to the model layers
for n in modified_weights_dict:
    w = modified_weights_dict[n]
    m = 0
    for num in original_weights_dict[n]:
        state_dict[n + '.weight'][num, :, :, :] = w[m, :, :, :]
        m += 1




# Recover the secret
# We store the extracted bits of data here
hidden_data: List[bool] = []

for n in layer_names:
    # Check if the layer was used in hiding the secret or not (e.g.: we could hide the secret in the prev. layers)
    if n not in last_index_used_in_layer_dict.keys():
        continue

    # We could get the modified weights directly from the model: model.get_layer(n).get_weights()...
    w = modified_weights_dict[n]
    w_shape = w.shape
    w = w.ravel()

    nb_params_in_layer: int = np.prod(w.shape)

    for i in range(last_index_used_in_layer_dict[n] + 1):
        x = FloatBinary(w[i])
        hidden_bits = x.fraction[-BITS_TO_USE:]
        hidden_data.extend(hidden_bits)

    print(f"Layer {n} is processed, bits are extracted")

recovered_message: str = bits_to_str(list(map(int, hidden_data)))

def hash_str(s: str) -> str:
    return hashlib.md5(s.encode("ascii")).hexdigest()

if hash_str(recovered_message) == hash_str(secret_to_hide):
    print("Successful secret hiding and recovery! 🥳")
else:
    print("Recovered message is not the same as the original one 🤨")