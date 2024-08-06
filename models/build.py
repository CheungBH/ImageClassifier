from .models import CNNModel
import torch
import torch.nn as nn
import os


freeze_pretrain = {"mobilenet": [155, "classifier"],
                   "shufflenet": [167, "fc"],
                   "mnasnet": [155, "classifier"],
                   "resnet18": [59, "fc"],
                   "squeezenet": [49, "classifier"],
                   "resnet34": [107, "fc"],
                   }


class ModelBuilder:
    def __init__(self, pretrain=False):
        self.pretrain = pretrain
        # self.device = device

    def build(self, cls_num, backbone, device):
        self.device = device
        self.CNN = CNNModel(cls_num, backbone, load_pretrain=self.pretrain, device=self.device)
        if self.device != "cpu":
            self.CNN.model.cuda()
        self.params_to_update = self.CNN.model.parameters()
        self.sigmoid = torch.nn.Sigmoid()
        return self.CNN.model

    def load_weight(self, path):
        self.CNN.load(path)
    
    def build_with_args(self, args):
        self.device = args.device
        self.backbone = args.backbone
        self.cls_num = args.cls_num
        self.device = args.device
        model = self.build(self.cls_num, self.backbone, self.device)
        if args.load_weight:
            self.load_weight(args.load_model)
        if args.freeze:
            self.freeze(args.freeze)
        os.makedirs(args.save_dir, exist_ok=True)
        self.write_structure(os.path.join(args.save_dir, "model.txt"), model)
        return model

    def get_benchmark(self, input_size=224):
        return self.CNN.get_benchmark(input_size)

    def inference(self, img_tns):
        img_tensor_list = [torch.unsqueeze(img_tns, 0)]
        input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
        self.image_batch_tensor = input_tensor.cuda() if self.device != "cpu" else input_tensor
        outputs = self.CNN.model(self.image_batch_tensor)
        outputs_tensor = outputs.data
        # m_softmax = nn.Sig(dim=1)
        outputs_tensor = self.sigmoid(outputs_tensor).to("cpu")
        return outputs_tensor

    def freeze(self, freeze, freeze_bn=0):
        try:
            feature_layer_num = freeze_pretrain[self.backbone][0]
            classifier_layer_name = freeze_pretrain[self.backbone][1]
            feature_num = int(freeze * feature_layer_num)

            for idx, (n, p) in enumerate(self.CNN.model.named_parameters()):
                if len(p.shape) == 1 and freeze_bn:
                    p.requires_grad = False
                elif classifier_layer_name not in n and idx < feature_num:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
        except:
            raise ValueError("This model is not supported for freezing now")

    @staticmethod
    def write_structure(file_path, model):
        print(model, file=open(file_path, "w"))




