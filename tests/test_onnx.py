import sys
sys.path.append('./python')
import numpy as np
import needle as ndl
import needle.nn as nn

import needle.onnx_dict as onnx_dict
import needle.onnx_parser as onnx_parser
sys.path.append('.')
from apps.models import ModelFromOnnx

import onnx
import onnxruntime
from onnx2torch import convert
import torch

'''ResNet Test'''
# onnx_model = onnx.load("models/resnet50.onnx")
# node_list = onnx_model.graph.node
# initializer_list = onnx_model.graph.initializer
# init_dict = onnx_parser.load_initializer(initializer_list)
# onnx_node_list = onnx_parser.load_node(node_list,init_dict)
# assert len(onnx_node_list) == len(node_list)

# # inferece Needle model transferred from Onnx
# device = ndl.cpu()
# model = ModelFromOnnx(onnx_node_list, device=device)
# test_input_np = np.random.rand(1, 3, 224, 224).astype('float32')
# ndl_input = ndl.Tensor(ndl.NDArray(test_input_np), device=device)
# test_out = model(ndl_input)

# # inference Onnx model
# session = onnxruntime.InferenceSession("models/resnet50.onnx")
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name
# onnx_out = session.run([output_name], {input_name: test_input_np})
# onnx_out_np = np.squeeze(np.array(onnx_out), axis=0)
# assert test_out.shape == onnx_out_np.shape
# # print(np.max(np.abs(test_out.numpy() - onnx_out_np)))
# assert np.allclose(test_out.numpy(), onnx_out_np, atol=1e-05)

# # inference Torch model transferred from Onnx
# torch_model = convert(onnx_model)
# torch_input = torch.tensor(test_input_np)
# torch_out = torch_model(torch_input)
# # print(torch_out.shape)
# # print(torch_out[0, :10])


'''RNN Test'''
onnx_model = onnx.load("models/rnn.onnx")
node_list = onnx_model.graph.node
initializer_list = onnx_model.graph.initializer
init_dict = onnx_parser.load_initializer(initializer_list)
onnx_node_list = onnx_parser.load_node(node_list, init_dict)

# inferece Needle model transferred from Onnx
device = ndl.cpu()
model = ModelFromOnnx(onnx_node_list, device=device)
test_input_np = np.random.randint(0, 16, (5, 1)).astype('long')
ndl_input = ndl.Tensor(ndl.NDArray(test_input_np), device=device)
test_out = model(ndl_input)

# inference Onnx model
session = onnxruntime.InferenceSession("models/rnn.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
onnx_out = session.run([output_name], {input_name: test_input_np})
onnx_out_np = np.squeeze(np.array(onnx_out), axis=0)

assert test_out.shape == onnx_out_np.shape
print(np.max(np.abs(test_out.numpy() - onnx_out_np)))
assert np.allclose(test_out.numpy(), onnx_out_np, atol=1e-05)
