"""Parse onnx model to onnx_dict"""
import onnx
import numpy as np
from .onnx_dict import *
import onnx.numpy_helper
# initializer mostly store value and dims of weight/bias
def load_initializer(initializer_list):
    init_dict={}
    for init in initializer_list:
        # pack initializer into object
        dims=init.dims
        dtype=init.data_type
        # import pdb; pdb.set_trace()
        data=onnx.numpy_helper.to_array(init).reshape(dims)
        name=init.name
        init_object=OnnxData(name=name,dtype=dtype,data=data,dims=dims)
        init_dict[name]=init_object
    return init_dict

# node constain attribute and data_name of each operator
def load_node(node_list,init_dict):
    onnx_dict_list=[]
    constant_dict={}
    for node in node_list:
        attributes=list(node.attribute)
        if len(node.input)==0:
            if node.op_type=='Constant':
                # get attributes
                assert len(attributes)==1
                constant_dict[node.output[0]]=onnx.numpy_helper.to_array(attributes[0].t)
            
            continue
        att_dict={'inputs':[node.input[0]],'output':[node.output[0]],'name':node.name}
        
        if node.op_type=='Conv':
            # get data field
            att_dict['X_name']=node.input[0]
            att_dict['W_name']=node.input[1]
            att_dict['B_name']=node.input[2]
            att_dict['W_data']=init_dict[att_dict['W_name']]
            att_dict['B_data']=init_dict[att_dict['B_name']]
            att_dict['Y_name']=node.output[0]
            # get attributes
            for att in attributes:
                if att.name=='dilations':
                    att_dict['dilations']=att.ints
                if att.name=='group':
                    att_dict['group']=att.i
                if att.name=='kernel_shape':
                    att_dict['kernel_shape']=att.ints
                if att.name=='pads':
                    att_dict['pads']=att.ints
                if att.name=='strides':
                    att_dict['strides']=att.ints
            conv_object=ConvOpNode(att_dict)
            onnx_dict_list.append(conv_object)
        if node.op_type=='BatchNormalization':
            # get data field
            att_dict["X_name"]=node.input[0]
            att_dict["gamma_name"]=node.input[1]
            assert 'gamma' in att_dict['gamma_name']
            att_dict["beta_name"]=node.input[2]
            assert 'beta' in att_dict['beta_name']
            att_dict["running_mean_name"]=node.input[3]
            assert 'running_mean' in att_dict['running_mean_name']
            att_dict["running_var_name"]=node.input[4]
            assert 'running_var' in att_dict['running_var_name']
            att_dict["Y_name"]=node.output[0]
            att_dict["gamma_data"]=init_dict[att_dict["gamma_name"]]
            att_dict["beta_data"]=init_dict[att_dict["beta_name"]]
            att_dict["running_mean_data"]=init_dict[att_dict["running_mean_name"]]
            att_dict["running_var_data"]=init_dict[att_dict["running_var_name"]]
            # get attributes
            for att in attributes:
                if att.name=='epsilon':
                    att_dict["epsilon"]=att.f
                if att.name=='momentum':
                    att_dict["momentum"]=att.f
                if att.name=='spatial':
                    att_dict["spatial"]=att.i
            BN_object=BatchNorm2DNode(att_dict)
            onnx_dict_list.append(BN_object)
        if node.op_type=='Relu':
            # get data field
            att_dict["X_name"]=node.input[0]
            att_dict["Y_name"]=node.output[0]
            ReLU_object=ReLUNode(att_dict)
            onnx_dict_list.append(ReLU_object)
        if node.op_type=='MaxPool':
            # get data field
            att_dict["X_name"]=node.input[0]
            att_dict["Y_name"]=node.output[0]
            # get attributes
            for att in attributes:
                if att.name=='kernel_shape':
                    att_dict['kernel_shape']=att.ints
                if att.name=='pads':
                    att_dict['pads']=att.ints
                if att.name=='strides':
                    att_dict['strides']=att.ints
            MaxPool_object=MaxPoolNode(att_dict)
            onnx_dict_list.append(MaxPool_object)
        if node.op_type=='Add': 
            # get data field
            att_dict['inputs']=node.input[:2]
            att_dict["A_name"]=node.input[0]
            att_dict["B_name"]=node.input[1]                   
            att_dict["C_name"]=node.output[0] 
            Add_object=AddNode(att_dict)
            onnx_dict_list.append(Add_object)
                                
        if node.op_type=='GlobalAveragePool':
            # get data field
            att_dict["X_name"]=node.input[0]
            att_dict["Y_name"]=node.output[0]
            GlobalAvgPool_object=GlobalAvgPoolNode(att_dict)
            onnx_dict_list.append(GlobalAvgPool_object)
        if node.op_type=='Flatten':
            # get data field
            att_dict["input_name"]=node.input[0]
            att_dict["output_name"]=node.output[0]
            Flatten_object=FlattenNode(att_dict)
            onnx_dict_list.append(Flatten_object)
        # linear layer
        if node.op_type=='Gemm':
            # get data field
            att_dict["W_name"]=node.input[1]
            assert 'weight' in att_dict['W_name']
            att_dict["B_name"]=node.input[2]
            assert 'bias' in att_dict['B_name']
            att_dict["W_data"]=init_dict[att_dict["W_name"]]
            att_dict["B_data"]=init_dict[att_dict["B_name"]]
            # get attributes
            for att in attributes:
                if att.name=='alpha':
                    att_dict['alpha']=att.f
                if att.name=='beta':
                    att_dict['beta']=att.f
                if att.name=='transA':
                    att_dict['transA']=att.i
                if att.name=='transB':
                    att_dict['transB']=att.i
            Gemm_object=GemmNode(att_dict)
            onnx_dict_list.append(Gemm_object)
        # embedding layer
        if node.op_type=='Gather' and 'embedding' in node.name:
            # import pdb; pdb.set_trace()
            # get data field
            att_dict["inputs"]=[node.input[1]]
            att_dict["X_name"]=node.input[1]
            att_dict["W_name"]=node.input[0]
            # att_dict["Y_name"]
            att_dict["W_data"]=init_dict[att_dict["W_name"]]
            weight_shape=att_dict["W_data"].data.shape
            att_dict['num_embeddings']=weight_shape[0]
            att_dict['embedding_dim']=weight_shape[1]
            Embedding_object=EmbeddingNode(att_dict)
            onnx_dict_list.append(Embedding_object)
        # RNN
        if node.op_type=='RNN':
            # enforce RNN to connect directly with embedding to match Needle
            assert att_dict['inputs']==['/embedding/Gather_output_0']
            att_dict["X_name"]=att_dict['inputs'][0]
            att_dict["W_ih_name"]=node.input[1]
            att_dict["W_hh_name"]=node.input[2]
            att_dict["B_name"]=node.input[3]
            att_dict["W_ih_data"]=init_dict[att_dict["W_ih_name"]]
            att_dict["W_hh_data"]=init_dict[att_dict["W_hh_name"]]
            att_dict["B_data"]=init_dict[att_dict["B_name"]]
            att_dict['output']=node.output
            # get attributes
            for att in attributes:
                if att.name=='activations':
                    att_dict['activations']=att.strings[0].decode("utf-8")
                if att.name=='hidden_size':
                    att_dict['hidden_size']=att.i
            
            att_dict["Y_name"]=att_dict['output'][0]
            att_dict["Y_h_name"]=att_dict['output'][1]
            RNN_object=RNNNode(att_dict)
            onnx_dict_list.append(RNN_object)
        if node.op_type=='Squeeze':
            att_dict["axes"]=constant_dict[node.input[1]].tolist()[0]
            att_dict['data']=node.input[0]
            att_dict["squeezed"]=node.output[0]
            Squeeze_object=SqueezeNode(att_dict)
            onnx_dict_list.append(Squeeze_object)
        if node.op_type=='Reshape':
            att_dict["data"]=node.input[0]
            att_dict["shape"]=constant_dict[node.input[1]]
            att_dict["reshaped"]=node.output[0]
            # get attributes
            for att in attributes:
                if att.name=='allowzero':
                    att_dict['allowzero']=att.i
            Reshape_object=ReshapeNode(att_dict)
            onnx_dict_list.append(Reshape_object)
    return onnx_dict_list

def main(model_path):
    onnx_model=onnx.load(model_path)
    node_list=onnx_model.graph.node
    initializer_list=onnx_model.graph.initializer
    init_dict=load_initializer(initializer_list)
    onnx_dict_list=load_node(node_list,init_dict)
    return onnx_dict_list,node_list

if __name__ == '__main__':
    model_path="/Users/elvisshi/Desktop/rnn.onnx"
    # model_path="/Users/elvisshi/Desktop/resnet18.onnx"
    onnx_dict_list,node_list=main(model_path)
    if 'rnn' not in model_path:
        # we need to skip some node in RNN to match Needle implementation.
        assert len(onnx_dict_list)==len(node_list)
