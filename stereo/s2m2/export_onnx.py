import argparse
import os, sys


import numpy as np
import cv2
import torch
import torch._dynamo

project_root = os.path.abspath('../../thirdparty/s2m2')
sys.path.append(os.path.join(project_root, 'src'))
print(project_root)


def export_onnx(model, onnx_path, left_torch, right_torch):

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    model = model.cpu().eval()
    left_torch, right_torch = left_torch.cpu(), right_torch.cpu()
    try:
        torch.onnx.export(model,
                          (left_torch, right_torch),
                          onnx_path,
                          export_params=True,
                          dynamo=False,
                          opset_version=18,
                          verbose=True,
                          do_constant_folding=False,
                          input_names=['input_left', 'input_right'],
                          output_names=['output_disp', 'output_occ', 'output_conf'],
                          dynamic_axes=None)
        print('success onnx conversion')

    except Exception as e:
        print(f"error type:{type(e).__name__}")
        print(f"error :{e}")
        import traceback
        traceback.print_exc()


def export_torchscript(model, torchscript_path, left_torch, right_torch):

    os.makedirs(os.path.dirname(torchscript_path), exist_ok=True)
    try:
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                exported_mod = torch.export.export(model, (left_torch, right_torch))

        torch.export.save(exported_mod, torchscript_path)
        print('success torchscript conversion')

    except Exception as e:
        print(f"error type:{type(e).__name__}")
        print(f"error :{e}")
        import traceback
        traceback.print_exc()




from s2m2.core.utils.model_utils import load_model
from s2m2.core.utils.image_utils import read_images
from s2m2.core.utils.vis_utils import visualize_stereo_results_2d
# from s2m2.tools.export_model import export_onnx
import onnxruntime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch._dynamo.config.verbose = True
torch.manual_seed(0)
np.random.seed(0)

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='S', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    parser.add_argument('--img_height', default=1088, type=int,
                        help='image height')
    parser.add_argument('--img_width', default=800, type=int,
                        help='image width')

    return parser

def main(args):
    img_height, img_width = args.img_height, args.img_width

    model_path = r"C:\Users\86994\Desktop\pretrain_weights_s2m2"
    model = load_model(model_path, args.model_type, not args.allow_negative, args.num_refine, 'cpu')

    # load web stereo images

    left_path = r"C:\Users\86994\Desktop\camera_kit\stereo\calibration\imgs_left_rectify\WIN_20251207_14_54_12_Pro.png"
    right_path = r"C:\Users\86994\Desktop\camera_kit\stereo\calibration\imgs_right_rectify\WIN_20251207_14_54_12_Pro.png"

    # load stereo images
    left, right = read_images(left_path, right_path)

    print(f"image size: {img_height}, {img_width}")

    if left.shape[1]>=img_width and left.shape[0]>=img_height:
        left = left[:img_height, :img_width]
        right = right[:img_height, :img_width]
    else:
        left = cv2.resize(left, dsize=(img_width, img_height))
        right = cv2.resize(right, dsize=(img_width, img_height))


    # to torch tensor
    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0))
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0))

    print(right_torch.shape, right_torch.max())

    torch_version = torch.__version__
    onnx_path = os.path.join(model_path, 'onnx_save', f'S2M2_{args.model_type}_{img_width}_{img_height}_v2_torch{torch_version[0]}{torch_version[2]}.onnx')
    print(onnx_path)

    print("开始导出.................................................................................")
    export_onnx(model, onnx_path, left_torch, right_torch)

    print("导出完毕....................................................")
    # test onnx file with onnxruntime in gpu
    sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    print(f"onnxruntime device: {onnxruntime.get_device()}")

    input_name = [input.name for input in sess.get_inputs()]
    output_name = [output.name for output in sess.get_outputs()]
    print(f"onnx_input_name:{input_name}")
    print(f"onnx_output_name:{output_name}")
    outputs = sess.run([output_name[0], output_name[1], output_name[2]],
                          {input_name[0]: left_torch.numpy(),
                           input_name[1]: right_torch.numpy()})
    print(f"onnx_output shape: {outputs[1].shape}")

    pred_disp, pred_occ, pred_conf = outputs
    pred_disp = np.squeeze(pred_disp)
    pred_occ = np.squeeze(pred_occ)
    pred_conf = np.squeeze(pred_conf)

    # opencv 2D visualization
    visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)