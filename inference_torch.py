#%%
from PIL import Image
import torch
from preprocess.converter import NormalizePAD, TokenLabelConverter
from models.model import Model, ViTSTR
import io
# %%
def preprocess_image(path, mode):
    data_transforms = NormalizePAD((1, 224, 224))
    if mode == 'api':
        img = Image.open(io.BytesIO(path)).convert('L')
    else:
        img = Image.open(path).convert('L')
    img = img.resize((224, 224), Image.BICUBIC)
    img = data_transforms(img)
    img = img.unsqueeze(0)
    return img

def prediction(model, path, converter, mode):
    model.eval()
    img = preprocess_image(path, mode)
    with torch.no_grad():
        preds = model(img, seqlen=converter.batch_max_length)
        _, pred_index = preds.topk(1, dim=-1, largest=True, sorted=True)
        pred_index = pred_index.view(-1, converter.batch_max_length)
        length_for_pred = torch.IntTensor([converter.batch_max_length - 1])
        pred_str = converter.decode(pred_index[:, 1:], length_for_pred)
        pred_EOS = pred_str[0].find('[s]')
        pred_str = pred_str[0][:pred_EOS]
    return pred_str
#%%
if __name__ == '__main__':
    path = 'examples/16.jpg'
    model = torch.load('./models/last_model.pth')
    converter = TokenLabelConverter()
    pred_str = prediction(model, path, converter, 'x')
    print(pred_str)
# %%
