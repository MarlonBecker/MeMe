from typing import List, Tuple, Union, Optional
import torch
import numpy as np

class LocallyConnectedNet(torch.nn.Module):
    def __init__(self,
                 action_space_shape,
                 model_config: dict,
                    ):
        torch.nn.Module.__init__(self)

        # save general parameters
        (self.w, self.h) = action_space_shape


        self.bias = model_config["bias"]

        # prepare other variables
        self._features = None

        # self._auto_calc_architecture(channel_proposals = None)
        self._calculate_dense_architecture(layers = model_config["layers"])

        if model_config["networkInitStructure"] is not None:
            learnedLayerInitStructure = np.loadtxt(model_config["networkInitStructure"], delimiter = ",", dtype = float)
            learnedLayerInitStructure = -1*(learnedLayerInitStructure*2-1)*model_config["networkInitScale"]
            if tuple(action_space_shape) != tuple(learnedLayerInitStructure.shape):
                raise RuntimeError(f"Expected 'networkInitStructure' of shape {action_space_shape} but got shape {learnedLayerInitStructure.shape}.")
            paddingWidth = (self.learned_layer_shape[1]-learnedLayerInitStructure.shape[0])//2
            learnedLayerInitStructure = np.expand_dims(np.pad(learnedLayerInitStructure, paddingWidth, mode = "linear_ramp", end_values=0), axis=0)
        else:
            learnedLayerInitStructure = None
        self.learnedLayer = LearnableParameterLayer(self.learned_layer_shape, initStructure=learnedLayerInitStructure)


        """ LOCALLY CONNECTED LAYERS """
        lcnet_layers = torch.nn.ModuleList([])
        for i in range(len(self.local_fc_filters)):
            if i == 0:
                input_channel  = self.learned_layer_shape[0]
                input_shape  = self.learned_layer_shape[1:]
            else:
                input_channel  = self.local_fc_filters[i-1][0] 
                input_shape  = output_shape

            output_channel = self.local_fc_filters[i][0] if i != len(self.local_fc_filters)-1 else 1

            lcnet_layers.append(
                LocalFCLayer2D(
                    input_shape    = input_shape,
                    kernel_size    = self.local_fc_filters[i][1],
                    stride         = self.local_fc_filters[i][2],
                    input_channel  = input_channel,
                    output_channel = output_channel,
                    bias           = self.bias,
                    boundary         = self.local_fc_filters[i][3]
                    )
                )

            output_shape = lcnet_layers[-1].get_output_shape()

        self.layers = torch.nn.Sequential(
            self.learnedLayer,
            *lcnet_layers,
        )
        
        if model_config["freezeLC"]:
            for layer in self.layers:
                if isinstance(layer, LocalFCLayer2D):
                    layer.weights.requires_grad = False

    def forward(self, input_ = None):
        out = self.layers(input_)

        return out.flatten()


    def _calculate_dense_architecture(self, kernel_size: int = 2, layers: int = 10, stride: int = 1, channels: int = 1) -> None:

        self.learned_layer_shape = [channels, self.w + layers * (kernel_size-1), self.h + layers * (kernel_size-1)]
        self.local_fc_filters = [[channels, [kernel_size, kernel_size], [stride, stride], "shrink" ]]*layers


    def _auto_calc_architecture(self, channel_proposals: Optional[List[int]] = None, silent = False) -> None:
        """auto generating architecture for stride=2 until min_dim is reached
            channel_proposals:
                list of channels in reverse order
        """
        
        stride = 1
        min_dim = 92

        # prefered kernel size per layer (starting at the end)
        prefered_kernel_size_even = [2, 2] + [2] * 100
        prefered_kernel_size_odd  = [3] * 100
        if channel_proposals is not None:
            channels = channel_proposals + [1] * 100
        else:
            channels = [1] * 100

        self.local_fc_filters = []

        if not silent:
            print("--> Auto calculating architecture for LocallyConnectedNet <--")
        dim_x, dim_y = self.w, self.h
        s_x  , s_y   = stride, stride
        i = 0
        while dim_x > min_dim or dim_y > min_dim:
            if dim_x > min_dim:
                k_x = prefered_kernel_size_even[i] if dim_x % 2 == 0 else prefered_kernel_size_odd[i]
            else:
                k_x = 1
                s_x = 1

            if dim_y > min_dim:
                k_y = prefered_kernel_size_even[i] if dim_y % 2 == 0 else prefered_kernel_size_odd[i]
            else:
                k_y = 1
                s_y = 1

            self.local_fc_filters.append(
                [
                    channels[i], #channels
                    [k_x, k_y] , #kernel size
                    [s_x, s_y] , #stride
                    "extend", #boundary mode
                ]
            )

            dim_x = (dim_x - k_x) // s_x + 1
            dim_y = (dim_y - k_y) // s_y + 1

            i += 1

        self.learned_layer_shape = [1, dim_x, dim_y]
        self.local_fc_filters = self.local_fc_filters[::-1]

        if not silent:
            print(f"learned layer dim: {dim_x}, {dim_y}")
            print(f"locally connected layers:")
            for i, layer in enumerate(self.local_fc_filters):
                dim_x = (dim_x - 1) * layer[2][0] + layer[1][0]
                dim_y = (dim_y - 1) * layer[2][1] + layer[1][1]
                print(f"{i}: channels = {layer[0]}; kernel = ({layer[1][0]}, {layer[1][1]}); stride = ({layer[2][0]}, {layer[2][1]}) -> dim: ({dim_x}, {dim_y})")



class LearnableParameterLayer(torch.nn.Module):
    def __init__(self, shape, initStructure = None):
        super(LearnableParameterLayer, self).__init__()
        self.shape = shape

        if initStructure is None:
            self.params = torch.nn.Parameter(torch.zeros(self.shape))
        else:
            self.params = torch.nn.Parameter(torch.from_numpy(initStructure))
            

    def forward(self, input_ = None):
        return self.params.unsqueeze(0)


class LocalFCLayer2D(torch.nn.Module):
    """ locally fully connected layer"""
    def __init__(self, input_shape: Tuple[int, int], kernel_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], input_channel: int, output_channel: int, bias: bool = True, boundary: str = "extend"):
        super().__init__()
        """
            boundary: "extend": output boundary neurons habe fewer connections, input are always connected; "shrink" all output boundary neurons have the same number if connections as inputs, but some input neurons are less connected  
        """

        self.input_channel : int = input_channel
        self.output_channel: int = output_channel
        self.input_shape: Tuple[int, int] = input_shape
        self.use_bias: bool = bias
        self.boundary: str = boundary

        if boundary not in ["shrink", "extend"]:
            raise ValueError(f"bondray should be either 'shrink' or 'extend' not '{boundary}'.")

        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size: Tuple[int, int] = tuple(kernel_size)
        elif isinstance(kernel_size, int):
            self.kernel_size: Tuple[int, int] = (kernel_size, kernel_size)
        else:
            raise ValueError(f"kernel_size must be int or tuple[int, int] but is {kernel_size} ({type(kernel_size)})")

        if isinstance(stride, (list, tuple)):
            self.stride: Tuple[int, int] = tuple(stride)
        elif isinstance(stride, int):
            self.stride: Tuple[int, int] = (stride, stride)
        else:
            raise ValueError(f"stride must be int or tuple[int, int] but is {stride} ({type(stride)})")

        assert self.kernel_size[0] >= self.stride[0]
        assert self.kernel_size[1] >= self.stride[1]


        if boundary == "extend":
            self.output_shape: Tuple[int, int] = (
                (self.input_shape[0] - 1) * self.stride[0] + self.kernel_size[0],
                (self.input_shape[1] - 1) * self.stride[1] + self.kernel_size[1]
                )
        elif boundary == "shrink":
            if not self.stride[0] == 1 and self.stride[1] == 1:
                raise ValueError(f"Boundary mode 'shrink' only supported for stride=1 atm. (stride is: {self.stride})")
            self.output_shape: Tuple[int, int] = (
                self.input_shape[0] + 1 - self.kernel_size[0],
                self.input_shape[1] + 1 - self.kernel_size[1],
                )

        
        # define params
        self.weights = torch.nn.Parameter(1/(self.kernel_size[0]*self.kernel_size[1])*torch.ones((self.input_channel, self.output_channel, *self.input_shape, *self.kernel_size)))
        self.bias = torch.nn.Parameter(torch.zeros((self.output_channel, *self.output_shape))) if self.use_bias else None
        # self.weights = torch.nn.Parameter(torch.zeros((self.input_channel, self.output_channel, *self.input_shape, *self.kernel_size)))
        # self.bias = torch.nn.Parameter(torch.zeros((self.output_channel, *self.output_shape))) if self.use_bias else None

    def get_output_shape(self):
        return self.output_shape

    def forward(self, x):
        # print(torch.max(self.weights), torch.min(self.weights), torch.max(self.weights) - torch.min(self.weights))
        if self.boundary == "extend":
            #b -> batch; i,j -> spatial dim; r,s -> kernel spatial dim; l,m input/output channel
            y = torch.einsum("blij, lmijrs -> bmrsij", x, self.weights)

            out = torch.zeros((x.shape[0], self.output_channel, *self.output_shape))
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[1]):
                    out[
                        :,
                        :,
                        i:(self.output_shape[0]-self.kernel_size[0]+i+1):self.stride[0],
                        j:(self.output_shape[1]-self.kernel_size[1]+j+1):self.stride[1]
                        ] += y[:, :, i, j, :, :]
            return out + self.bias if self.use_bias else out

        if self.boundary == "shrink":
            #b -> batch; i,j -> spatial dim; r,s -> kernel spatial dim; l,m input/output channel
            y = torch.einsum("blij, lmijrs -> bmrsij", x, self.weights)
            out = torch.zeros((x.shape[0], self.output_channel, *self.output_shape))
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[1]):
                    out += y[
                             :,
                             :,
                             i,
                             j,
                             self.kernel_size[0]-1-i:self.input_shape[0]-i,
                             self.kernel_size[1]-1-j:self.input_shape[1]-j,
                                 ]
            return out + self.bias if self.use_bias else out
