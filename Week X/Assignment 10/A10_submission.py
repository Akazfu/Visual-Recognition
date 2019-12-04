import numpy as np
import cv2 as cv
import torch
# from model import *

# Modified predict(num_classes) function for DenseNet
class DenseNet :
    def __init__(self,pth,mode="cpu"):
        self.device = torch.device(mode)
        self.DenseNet = torch.load(pth, map_location = torch.device(mode))

    def predict(self,image):
        # Set for eval mode, require grad = False
        self.DenseNet.eval()
        gray = cv.cvtColor(image.reshape(64, 64), cv.COLOR_GRAY2BGR)

        with torch.no_grad():
            gray_tensor = torch.from_numpy(gray.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).to(self.device)
            # Onehot vector of size 28 for the output layer, 10 for first digit, 10 for secon digit, and 8 for the bbox
            oh = self.DenseNet(gray_tensor)
            oh_class = oh[:, :20].contiguous().view(-1, 10)
            oh_box = oh[:, 20:]

            # Sort the tensor by ascending order
            pred_class = oh_class.argmax(1).cpu().numpy()
            pred_box = oh_box.long().cpu().numpy()[0].reshape(2,4)

        return pred_class,pred_box

def classify_and_detect(images):
    """

    :param np.ndarray images: N x 4096 array containing N 64x64 images flattened into vectors
    :return: np.ndarray, np.ndarray
    """
    N = images.shape[0]

    # pred_class: Your predicted labels for the 2 digits, shape [N, 2]
    pred_class = np.empty((N, 2), dtype=np.int32)
    # pred_bboxes: Your predicted bboxes for 2 digits, shape [N, 2, 4]
    pred_bboxes = np.empty((N, 2, 4), dtype=np.float64)

    # add your code here to fill in pred_class and pred_bboxes
    model = DenseNet("checkpoint_17.pth")

    for i in range(N):
        label,box=model.predict(images[i,:])
        pred_class[i,:]=label
        pred_bboxes[i,:]=box

        if i % 100 == 0:
            print('Evaluating: [{}/{} ({:.0f}%)]\n'.format(i, N, (i/N*100)))

    return pred_class, pred_bboxes
