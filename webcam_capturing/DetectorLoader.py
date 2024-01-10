from ResNetL import resnetl10
from torch import nn
import torch


def _construct_depth_model(base_model):
    # modify the first convolution kernels for Depth input
    modules = list(base_model.modules())

    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                 list(range(len(modules)))))[0]
    conv_layer = modules[first_conv_idx]
    container = modules[first_conv_idx - 1]

    # modify parameters, assume the first blob contains the convolution kernels
    motion_length = 1
    params = [x.clone() for x in conv_layer.parameters()]
    kernel_size = params[0].size()
    new_kernel_size = kernel_size[:1] + (1*motion_length,  ) + kernel_size[2:]
    new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

    new_conv = nn.Conv3d(1, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride,
                         conv_layer.padding, bias=True if len(params) == 2 else False)
    new_conv.weight.data = new_kernels
    if len(params) == 2:
        new_conv.bias.data = params[1].data # add bias if neccessary
    layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

    # replace the first convlution layer
    setattr(container, layer_name, new_conv)

    return base_model


def modify_kernels(model):
    print("[INFO]: Converting the pretrained model to Depth init model")
    model = _construct_depth_model(model)
    print("[INFO]: Done. Flow model ready.")
    modules = list(model.modules())
    first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv3d),
                                               list(range(len(modules)))))[0]
    #conv_layer = modules[first_conv_idx]
    #if conv_layer.kernel_size[0]> opt.sample_duration:
     #   model = _modify_first_conv_layer(model,int(opt.sample_duration/2),1)
    return model


model = resnetl10(
    # num_classes=opt.n_classes,
    # shortcut_type=opt.resnet_shortcut,
    # sample_size=opt.sample_size,
    # sample_duration=opt.sample_duration
    num_classes=2,
    shortcut_type='A',
    sample_size=112,
    sample_duration=8, 
    )

detector = modify_kernels(model)
parameters = model.parameters()

# # TO parameter
# if resume_path:
#     opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
#     print('loading checkpoint {}'.format(opt.resume_path))
#     checkpoint = torch.load(opt.resume_path)
#     #assert opt.arch == checkpoint['arch']

#     detector.load_state_dict(checkpoint['state_dict'])


print('Model 1 \n', detector)
pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                            p.requires_grad)
print("Total number of trainable parameters: ", pytorch_total_params)
