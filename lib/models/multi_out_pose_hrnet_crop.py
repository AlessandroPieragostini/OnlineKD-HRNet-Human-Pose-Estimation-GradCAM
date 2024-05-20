import torch.nn as nn
from models.pose_hrnet import PoseHighResolutionNet, BN_MOMENTUM, Bottleneck, blocks_dict

def get_pose_net(cfg, is_train, **kwargs):
    model = MultiOutPoseHrnet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model


class MultiOutPoseHrnet(PoseHighResolutionNet):
    
    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        self.cfg = cfg
        extra = cfg.MODEL.EXTRA
        super(PoseHighResolutionNet, self).__init__()
        # super(MultiOutPoseHrnet, self).__init__(cfg, **kwargs)

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        if cfg.MODEL.N_STAGE > 2:
            self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
            num_channels = self.stage3_cfg['NUM_CHANNELS']
            block = blocks_dict[self.stage3_cfg['BLOCK']]
            num_channels = [
                num_channels[i] * block.expansion for i in range(len(num_channels))
            ]
            self.transition2 = self._make_transition_layer(
                pre_stage_channels, num_channels)
            self.stage3, pre_stage_channels = self._make_stage(
                self.stage3_cfg, num_channels)

            if cfg.MODEL.N_STAGE > 3:
                self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
                num_channels = self.stage4_cfg['NUM_CHANNELS']
                block = blocks_dict[self.stage4_cfg['BLOCK']]
                num_channels = [
                    num_channels[i] * block.expansion for i in range(len(num_channels))
                ]
                self.transition3 = self._make_transition_layer(
                    pre_stage_channels, num_channels)
                self.stage4, pre_stage_channels = self._make_stage(
                    self.stage4_cfg, num_channels, multi_scale_output=False)
        
        
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']
        
        self.n_stage = cfg.MODEL.N_STAGE
        
        self.intermediate_layers = nn.ModuleList()
        
        for i in range(self.n_stage):
            self.intermediate_layers.append( nn.Conv2d(
            in_channels= cfg.MODEL.OUT_CHANNELS,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        ))
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        y_list = [x]
        
        stage_cfgs = [self.stage2_cfg]
        if self.cfg.MODEL.N_STAGE > 2:
            stage_cfgs.append(self.stage3_cfg)
            if self.cfg.MODEL.N_STAGE > 3:
                stage_cfgs.append(self.stage4_cfg)
        transitions = [self.transition1]
        if self.cfg.MODEL.N_STAGE > 2:
            transitions.append(self.transition2)
            if self.cfg.MODEL.N_STAGE > 3:
                transitions.append(self.transition3)
        stages = [self.stage2]
        if self.cfg.MODEL.N_STAGE > 2:
            stages.append(self.stage3)
            if self.cfg.MODEL.N_STAGE > 3:
                stages.append(self.stage4)
        outputs = []
        
        for index, (stage_cfg, transition, stage) in enumerate(zip(stage_cfgs, transitions, stages)):
            
            x_list = []
            for i in range(stage_cfg['NUM_BRANCHES']):
                if transition[i] is not None:
                    x_list.append(transition[i](y_list[-1]))
                else:
                    x_list.append(y_list[i])
            outputs.append(self.intermediate_layers[index](x_list[0]))
            
            if self.n_stage == index + 1:
                break
            y_list = stage(x_list)

        if self.n_stage == 4:
            outputs.append(self.final_layer(y_list[0]))
        
        return outputs
        