from ai_hub import inferServer
import json
import base64
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch.autograd import Variable as V
import base64
from model import UnetIBN
import torch.nn.functional as F


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.mean = torch.tensor([0.19014559, 0.23549616, 0.22207538, 0.52786024]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        self.std = torch.tensor([0.12677416, 0.12194977, 0.1188112, 0.19847064]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(device)
        data = torch.ones((1, 4, 256, 256)).to(device)
        self.model = model.to(device)
        checkpoint = torch.load('ibnUnet+fft+wc0005+dilation.pth',map_location=device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()


    # 数据前处理
    def pre_process(self, data):
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        bytesIO = BytesIO()
        img = np.asarray(Image.open(BytesIO(bytearray(img))),dtype=np.float32)
        return img


    # 数据后处理
    def post_process(self, data):
        img_encode = np.array(cv2.imencode('.png', data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data, 'utf-8')
        return bast64_str

    # 模型预测：默认执行self.model(preprocess_data)，一般不用重写
    # 如需自定义，可覆盖重写
    def predict(self, data):
        with torch.no_grad():
            img = torch.from_numpy(data.transpose(2, 0, 1)).unsqueeze(0).cuda()
            img = (img / 255. - self.mean) / self.std
            output = self.model (img)
            output = torch.argmax(output,dim=1)
            output = output+1
        pred = output.squeeze().cpu().data.numpy()
        pred = np.uint8(pred)
        return pred


if __name__ == "__main__":
    mymodel = UnetIBN(encoder_pretrained=False,classes=10)
    my_infer = myInfer(mymodel)
    my_infer.run(debuge=True)  # 默认为("127.0.0.1", 80)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息
