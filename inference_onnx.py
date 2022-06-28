#%%
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import torch
from preprocess.converter import NormalizePAD, TokenLabelConverter
from models.model import Model
#%%

def preprocess_img(path):
    data_transforms = NormalizePAD((1, 224, 224))
    img = Image.open(path).convert('L')
    img = img.resize((224, 224), Image.BICUBIC)
    img = data_transforms(img)
    img = img.unsqueeze(0)
    return img

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def predict_onnx(sess, path):
    input_name = sess.get_inputs()[0].name
    print(input_name)
    img = preprocess_img(path)
    preds = sess.run(None, {input_name: to_numpy(img)})
    preds = np.squeeze(preds)
    return preds
#%%
if __name__ == '__main__':
    sess = ort.InferenceSession('last_model.onnx')
    converter = TokenLabelConverter()
    path = 'examples/1.jpg'
    preds = predict_onnx(sess, path)
    print(preds)
#%%
    print(preds.shape)
    pred_index = preds.max(1)
    pred_index = pred_index.reshape(1, 25)
    print(pred_index)
    length_for_pred = np.array([25 - 1])
    pred_str = converter.decode(pred_index, length_for_pred)
    print(pred_str)
#%%
    
# %%
