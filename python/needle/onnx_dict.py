"""Data of each node type in Onnx"""
import numpy as np

class OnnxNode:
    def __init__(self, att_dict) -> None:
        self.name = att_dict['name']
        self.inputs = att_dict['inputs']
        self.indegree = len(self.inputs)
        for input_name in self.inputs:
            if "data" in input_name or "initial_" in input_name:
                self.indegree -= 1
        self.outputs = att_dict['output']


class OnnxData:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs['name']
        self.dtype = kwargs['dtype']
        self.category = "Initializer"   
        self.data: np.array = kwargs['data']
        self.dims: list = list(self.data.shape)


class ConvOpNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)
        
        # attribures
        self.dilations = att_dict["dilations"]
        self.group = att_dict["group"]
        self.kernel_shape = att_dict["kernel_shape"]
        self.padding = att_dict["pads"]
        self.strides = att_dict["strides"]
        
        # data field
        self.X_name = att_dict["X_name"]
        self.Y_name = att_dict["Y_name"]
        self.W_name = att_dict["W_name"]
        self.W: OnnxData = att_dict["W_data"]
        self.use_bias = False
        if "B_name" in att_dict:
            self.use_bias = True
            self.B_name = att_dict["B_name"]
            self.B: OnnxData = att_dict["B_data"]

        self.out_channels, self.in_channels, _, _ = self.W.dims


class BatchNorm2DNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # attribures
        self.eps = att_dict["epsilon"]
        self.momentum = att_dict["momentum"]
        self.spatial = att_dict["spatial"]

        # data field
        self.X_name = att_dict["X_name"]
        self.gamma_name = att_dict["gamma_name"]
        self.gamma: OnnxData = att_dict["gamma_data"]
        self.beta_name = att_dict["beta_name"]
        self.beta: OnnxData = att_dict["beta_data"]
        self.running_mean_name = att_dict["running_mean_name"]
        self.running_mean: OnnxData = att_dict["running_mean_data"]
        self.running_var_name = att_dict["running_var_name"]
        self.running_var: OnnxData = att_dict["running_var_data"]
        self.Y_name = att_dict["Y_name"]

        self.dim = self.gamma.dims[0]


class ReLUNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # data field
        self.X_name = att_dict["X_name"]
        self.Y_name = att_dict["Y_name"]


class TanhNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # data field
        self.X_name = att_dict["X_name"]
        self.Y_name = att_dict["Y_name"]


class MaxPoolNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)
        self.kernel_shape = att_dict["kernel_shape"]
        self.padding = att_dict["pads"]
        self.strides = att_dict["strides"]

        # data field
        self.X_name = att_dict["X_name"]
        self.Y_name = att_dict["Y_name"]


class AddNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # data field
        self.A_name = att_dict["A_name"]
        self.B_name = att_dict["B_name"]
        self.C_name = att_dict["C_name"]


class GlobalAvgPoolNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # data field
        self.X_name = att_dict["X_name"]
        self.Y_name = att_dict["Y_name"]


class FlattenNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # data field
        self.input_name = att_dict["input_name"]
        self.output_name = att_dict["output_name"]


class GemmNode(OnnxNode):
    """Y = alpha * A * B + beta * C"""
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # attribures
        self.alpha = att_dict["alpha"]
        self.beta = att_dict["beta"]
        # self.transA: bool = att_dict["transA"]
        self.transB: bool = att_dict["transB"]

        # data field
        self.W_name = att_dict["W_name"]
        self.W_data: OnnxData = att_dict["W_data"]
        self.B_name = att_dict["B_name"]
        self.B_data: OnnxData = att_dict["B_data"]

        self.out_dim, self.in_dim = self.W_data.dims


class EmbeddingNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # attribures
        self.num_embeddings = att_dict["num_embeddings"]  # Size of the dictionary
        self.embedding_dim = att_dict["embedding_dim"]  # Size of each embedding vector

        # data field
        self.X_name = att_dict["X_name"]
        # self.Y_name = att_dict["Y_name"]
        self.W_name = att_dict["W_name"]
        self.W: OnnxData = att_dict["W_data"]
        assert self.W.dims == [self.num_embeddings, self.embedding_dim]


class RNNNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # attribures
        self.hidden_size = att_dict["hidden_size"]
        self.activation: str = att_dict["activations"].lower()
        
        # data field
        self.X_name = att_dict["X_name"]
        self.h0_name = att_dict["initial_h"] if "initial_h" in att_dict else None
        self.Y_name = att_dict["Y_name"]
        self.Y_h_name = att_dict["Y_h_name"]
        self.W_ih_name = att_dict["W_ih_name"]
        self.W_ih: OnnxData = att_dict["W_ih_data"]
        self.W_hh_name = att_dict["W_hh_name"]
        self.W_hh: OnnxData = att_dict["W_hh_data"]
        self.B_name = att_dict["B_name"]
        self.B: OnnxData = att_dict["B_data"]
        
        assert self.W_ih.dims[1] == self.hidden_size
        self.input_size = self.W_ih.dims[2]
        assert self.W_hh.dims[1] == self.W_hh.dims[2] == self.hidden_size
        assert self.B.dims[1] == 2 * self.hidden_size


class ReshapeNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # attribures
        self.allowzero = att_dict["allowzero"]

        # data field
        self.X_name = att_dict["data"]
        self.Y_name = att_dict["reshaped"]
        self.new_shape = att_dict["shape"]


class SqueezeNode(OnnxNode):
    def __init__(self, att_dict) -> None:
        super().__init__(att_dict)

        # attribures
        self.axes = att_dict["axes"]

        # data field
        self.X_name = att_dict["data"]
        self.Y_name = att_dict["squeezed"]
