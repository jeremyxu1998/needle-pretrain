import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import needle.onnx_dict as onnx
import math
import numpy as np
np.random.seed(0)
import torch


class ModelFromOnnx(nn.Module):
    """Needle model that is built from a list of interconnected Onnx nodes"""
    def __init__(self, onnx_node_list, device=None, dtype="float32"):
        super().__init__()
        self.modules = []  # list of Needle modules in the model
        self.modules_input = []  # input variable names of each module
        self.onnx_output_list = []  # map node to output values

        onnx_input_dict = {}  # map input variable names to nodes
        next_node = []  # queue for next module to initialize
        for node in onnx_node_list:
            if node.indegree == 0:
                next_node.append(node)
            for node_input in node.inputs:
                if node_input in onnx_input_dict:
                    onnx_input_dict[node_input].append(node)
                else:
                    onnx_input_dict[node_input] = [node]

        while len(next_node) > 0:
            node = next_node.pop(0)
            # initialize the corresponding nn module from onnx node with attributes, construct and load weight tensors if needed
            if isinstance(node, onnx.ConvOpNode):
                next_module = nn.Conv(node.in_channels, node.out_channels, node.kernel_shape[0], 
                                      node.strides[0], bias=node.use_bias, device=device, dtype=dtype)
                if node.use_bias:
                    next_module.load_weights(ndl.Tensor(ndl.NDArray(node.W.data.transpose(2,3,1,0)), device=device, dtype=dtype),
                                             ndl.Tensor(ndl.NDArray(node.B.data), device=device, dtype=dtype))
                else:
                    next_module.load_weights(ndl.Tensor(ndl.NDArray(node.W.data.transpose(2,3,1,0)), device=device, dtype=dtype))
            elif isinstance(node, onnx.BatchNorm2DNode):
                next_module = nn.BatchNorm2d(node.dim, eps=node.eps, momentum=node.momentum, device=device, dtype=dtype)
                next_module.load_weights(ndl.Tensor(ndl.NDArray(node.gamma.data), device=device, dtype=dtype),
                                         ndl.Tensor(ndl.NDArray(node.beta.data), device=device, dtype=dtype),
                                         ndl.Tensor(ndl.NDArray(node.running_mean.data), device=device, dtype=dtype),
                                         ndl.Tensor(ndl.NDArray(node.running_var.data), device=device, dtype=dtype))
            elif isinstance(node, onnx.ReLUNode):
                next_module = nn.ReLU()
            elif isinstance(node, onnx.TanhNode):
                next_module = nn.Tanh()
            elif isinstance(node, onnx.MaxPoolNode):
                next_module = nn.MaxPool2d(node.kernel_shape[0], node.strides[0], node.padding[0])
            elif isinstance(node, onnx.AddNode):
                next_module = nn.Add()
            elif isinstance(node, onnx.GlobalAvgPoolNode):
                next_module = nn.GlobalAvgPool2d()
            elif isinstance(node, onnx.FlattenNode):
                next_module = nn.Flatten()
            elif isinstance(node, onnx.GemmNode):
                assert node.alpha == 1 and node.beta == 1
                next_module = nn.Linear(node.in_dim, node.out_dim, device=device, dtype=dtype)
                next_module.load_weights(ndl.Tensor(ndl.NDArray(node.W_data.data.T), device=device, dtype=dtype),
                                         ndl.Tensor(ndl.NDArray(np.expand_dims(node.B_data.data, axis=0)), device=device, dtype=dtype))
            elif isinstance(node, onnx.EmbeddingNode):
                next_module = nn.Embedding(node.num_embeddings, node.embedding_dim, device=device, dtype=dtype)
                next_module.load_weights(ndl.Tensor(ndl.NDArray(node.W.data), device=device, dtype=dtype))
            elif isinstance(node, onnx.RNNNode):
                next_module = nn.RNN(node.input_size, node.hidden_size, nonlinearity=node.activation, device=device, dtype=dtype)
                next_module.load_weights(W_ih=ndl.Tensor(ndl.NDArray(np.squeeze(node.W_ih.data).T), device=device, dtype=dtype),
                                         W_hh=ndl.Tensor(ndl.NDArray(np.squeeze(node.W_hh.data).T), device=device, dtype=dtype),
                                         b_ih=ndl.Tensor(ndl.NDArray(np.squeeze(node.B.data)[:node.hidden_size]), device=device, dtype=dtype),
                                         b_hh=ndl.Tensor(ndl.NDArray(np.squeeze(node.B.data)[node.hidden_size:]), device=device, dtype=dtype))
            elif isinstance(node, onnx.ReshapeNode):
                next_module = nn.Reshape(node.new_shape)
            elif isinstance(node, onnx.SqueezeNode):
                next_module = nn.Squeeze(axes=node.axes)

            self.modules.append(next_module)
            self.modules_input.append(node.inputs)

            self.onnx_output_list.append([])
            for node_out in node.outputs:
                self.onnx_output_list[-1].append(node_out)  # store output variable names

                for i, subsequent_node in enumerate(onnx_input_dict.get(node_out, [])):
                    # check all nodes that take current node's output as input
                    onnx_input_dict[node_out][i].indegree -= 1
                    if onnx_input_dict[node_out][i].indegree == 0:
                        next_node.append(subsequent_node)

    def forward(self, x):
        module_io_vals = {}  # store the output of each step
        for step, (module, input_ids) in enumerate(zip(self.modules, self.modules_input)):
            # construct module input from either module_io_vals (using variable name as key), or use model input
            inputs = [(module_io_vals[id] if not ("data" in id or "initial_" in id) else x) for id in input_ids]

            out = module(*inputs)

            if type(out) == tuple:  # a tuple of tensors (RNN case)
                assert len(self.onnx_output_list[step]) == len(out)
                for out_tensor_name, out_tensor in zip(self.onnx_output_list[step], out):
                    module_io_vals[out_tensor_name] = out_tensor
            else:  # single tensor
                assert len(self.onnx_output_list[step]) == 1
                module_io_vals[self.onnx_output_list[step][0]] = out

        return out


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        raise NotImplementedError() ###
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    # model = ResNet9()
    # x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    # model(x)
    # cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    # train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    # print(dataset[1][0].shape)

    empty_onnx_data_k_args = {"name": "input", "dtype": np.float32, "data": np.random.rand(4, 4, 3, 3), "dims": [4, 4, 3, 3]}
    empty_onnx_data_k = onnx.OnnxData(**empty_onnx_data_k_args)
    conv1_args = {"name": "conv1", "inputs": ["data"], "output": ["conv1_out"],
                  "dilations": (1, 1), "group": 1, "kernel_shape": (3, 3), "pads": (1,)*4, "strides": (1, 1),
                  "X_name": "data", "W_name": "conv1_weights", "Y_name": "conv1_out", "W_data": empty_onnx_data_k}
    conv1 = onnx.ConvOpNode(conv1_args)
    empty_onnx_data_args = {"name": "input", "dtype": np.float32, "data": np.random.rand(4), "dims": [4]}
    empty_onnx_data = onnx.OnnxData(**empty_onnx_data_args)
    bn1_args = {"name": "bn1", "inputs": ["conv1_out"], "output": ["bn1_out"],
                "epsilon": 1e-5, "momentum": 0.1, "spatial": 1,
                "X_name": "conv1_out", "Y_name": "bn1_out", "gamma_name": "gamma_1", "beta_name": "beta_1",
                "gamma_data": empty_onnx_data, "beta_data": empty_onnx_data,
                "running_mean_name": "bn1_rmean", "running_var_name": "bn1_rvar",
                "running_mean_data": empty_onnx_data, "running_var_data": empty_onnx_data}
    bn1 = onnx.BatchNorm2DNode(bn1_args)
    relu1_args = {"name": "relu1", "inputs": ["bn1_out"], "output": ["relu1_out"],
                  "X_name": "bn1_out", "Y_name": "relu1_out"}
    relu1 = onnx.ReLUNode(relu1_args)
    add1_args = {"name": "add1", "inputs": ["data", "relu1_out"], "output": ["add1_out"],
                 "A_name": "data", "B_name": "relu1_out", "C_name": "add1_out"}
    add1 = onnx.AddNode(add1_args)

    onnx_node_list = [bn1, add1, relu1, conv1]
    model = ModelFromOnnx(onnx_node_list, device=ndl.cpu())
    test_input_np = np.random.rand(1, 4, 5, 5).astype('float32')
    ndl_input = ndl.Tensor(ndl.NDArray(test_input_np), device=ndl.cpu())
    ndl_out = model(ndl_input)
    assert ndl_out.shape == (1, 4, 5, 5)
    # print(ndl_out.cached_data.numpy())

    torch_model = torch.nn.Sequential(
        torch.nn.Conv2d(4, 4, 3, padding='same', bias=False),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU())
    torch_model[0].weight.data = torch.tensor(model.modules[0].weight.cached_data.numpy().transpose(3, 2, 0, 1))
    torch_model[1].weight.data = torch.tensor(model.modules[1].weight.cached_data.numpy())
    torch_model[1].bias.data = torch.tensor(model.modules[1].bias.cached_data.numpy())
    torch_input = torch.tensor(test_input_np)
    torch_out = torch_model(torch_input) + torch_input
    # print(torch_out.data.numpy())

    print(np.linalg.norm(ndl_out.cached_data.numpy() - torch_out.data.numpy()))
