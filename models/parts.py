

def get_backbone(name, pretrain, use_dilation):
    if name in ['resnet18', 'resnet50', 'resnet101']:
        from vendor.monodepth2.resnet_encoder import ResnetEncoder
        from vendor.utils import get_resnet_num_layers, remove_resnet_tail_

        num_layers = get_resnet_num_layers(name)
        backbone = ResnetEncoder(num_layers,
                                 pretrained=pretrain if pretrain == 'imnet' else None,
                                 replace_stride_with_dilation=use_dilation)
        remove_resnet_tail_(backbone)
        return backbone, backbone.num_ch_enc

    raise NotImplementedError


def get_head(head, num_ch_enc, num_out_ch, **head_kwargs):
    if head == 'padnet+':
        from vendor.monodepth2.decoder import DecoderHead
        return DecoderHead(num_ch_enc, num_out_ch, **head_kwargs)

    elif head == 'deeplab':
        from vendor.mtlpt import DeepLabHead
        return DeepLabHead(num_ch_enc[-1], num_out_ch)

    raise NotImplementedError
