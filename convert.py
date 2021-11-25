#-*-coding:utf-8-*-

from models.build import ModelBuilder
import torch
import os


def convert(args):
    model_path = args.model_path
    backbone = args.backbone

    libtorch_path = args.libtorch_path
    onnx_path = args.onnx_path

    assert libtorch_path or onnx_path, "You should assign at least one type to convert!"

    num_cls = args.num_cls
    inp_size = args.inp_size

    MB = ModelBuilder()
    model = MB.build(num_cls, backbone)
    model.eval()
    MB.load_weight(model_path)

    dummy_input = torch.rand(2, 3, inp_size, inp_size).cuda()

    with torch.no_grad():
        if onnx_path:
            onnx_sim_path = onnx_path.replace(".onnx", "_sim.onnx")
            torch.onnx.export(model, dummy_input, onnx_path, verbose=False, )
            os.system("python -m onnxsim {} {}".format(onnx_path, onnx_sim_path))

        if libtorch_path:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(libtorch_path)


class AutoConvert:
    def __init__(self, output_folder):
        self.output_folder = output_folder

    def run(self, model_path, args):
        num_cls, inp_size, backbone = args
        output_dir = os.path.join(self.output_folder, model_path.split("/")[-2])
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, "model.onnx")
        libtorch_path = os.path.join(output_dir, "model.pt")
        cmd = "python convert.py --model_path {} --backbone {} --libtorch_path {} --onnx_path {} --num_cls {} " \
              "--inp_size {}".format(model_path, backbone, libtorch_path, onnx_path, num_cls, inp_size)
        os.system(cmd)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--backbone', required=True)

    parser.add_argument('--libtorch_path', default="", type=str)
    parser.add_argument('--onnx_path', default="", type=str)

    parser.add_argument('--num_cls', default=2, type=int)
    parser.add_argument('--inp_size', default=224, type=int)
    args = parser.parse_args()
    convert(args)




