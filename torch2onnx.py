from tabnanny import check
import torch
from torch import nn
import torch.nn.functional as F

from torch_lib.Model import Model
from torchvision.models import vgg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("run on", device)

# ----����vgg��Ϊģ�͵ĹǸ�����----
my_vgg = vgg.vgg19_bn(pretrained=True)
pytorch_model = Model(features=my_vgg.features, bins=2).to(device)
# ----------------------------
checkpoint = torch.load('weights/epoch_10.pkl', map_location=torch.device('cpu'))["model_state_dict"]
pytorch_model.load_state_dict(checkpoint)
pytorch_model.to(device)
pytorch_model.eval()
print("haved loaded model")

dummy_input = torch.zeros(1, 3, 224, 224).to(device)  # ������batch_size ͨ���� ͼƬ��С�� cause:vgg19's input need 224*224
in_names = ['input']
out_names = ['orient', 'conf', 'dim']
torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True,
                    input_names=in_names, output_names=out_names)
# torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', verbose=True)
print("Done!")