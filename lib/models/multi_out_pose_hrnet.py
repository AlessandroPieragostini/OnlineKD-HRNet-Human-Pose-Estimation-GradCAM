import torch.nn as nn
from models.pose_hrnet import PoseHighResolutionNet

def get_pose_net(cfg, is_train, **kwargs):
    model = MultiOutPoseHrnet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model

class MultiOutPoseHrnet(PoseHighResolutionNet):
    
    def __init__(self, cfg, **kwargs):
        extra = cfg.MODEL.EXTRA
        super(MultiOutPoseHrnet, self).__init__(cfg, **kwargs)
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
        
        stage_cfgs = [self.stage2_cfg, self.stage3_cfg, self.stage4_cfg]
        transitions = [self.transition1, self.transition2, self.transition3]
        stages = [self.stage2, self.stage3, self.stage4]
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
        