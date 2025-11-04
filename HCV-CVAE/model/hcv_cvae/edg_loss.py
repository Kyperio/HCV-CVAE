import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# --- Edge loss function  --- #
class Edg_Capture(nn.Module):
    def __init__(self):
        super(Edg_Capture, self).__init__()
        kernel = [[-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
 
    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


# --- Edge + CWMI Loss with dynamic weights, 保留函数名 edge_loss --- #
_edgecwmi_state = {
    'laplace': None,
    'L1': None,
    'w_edge': 1.0,
    'w_cwmi': 1.0,
    'history_edge': [],
    'history_cwmi': [],
    'num_classes': 21,
    'device': 'cuda'
}

def edge_loss(x, y, pred_logits=None, gt_labels=None, device='cuda', num_classes=21, epoch=None):
    """
    组合 Edge + CWMI Loss，保持原来的 edge_loss 函数名
    """
    # 初始化 state
    if _edgecwmi_state['laplace'] is None:
        _edgecwmi_state['laplace'] = Edg_Capture().to(device)
        _edgecwmi_state['L1'] = nn.L1Loss().to(device)
        _edgecwmi_state['num_classes'] = num_classes
        _edgecwmi_state['device'] = device

    laplace = _edgecwmi_state['laplace']
    L1 = _edgecwmi_state['L1']

    # --- Edge loss ---
    edge_val = L1(laplace(x), laplace(y))

    # --- CWMI loss ---
    if pred_logits is None or gt_labels is None:
        return edge_val  # 只计算 Edge Loss

    n, c, h, w = pred_logits.size()
    if (gt_labels.size(1) != h) or (gt_labels.size(2) != w):
        pred_logits = F.interpolate(pred_logits, size=(gt_labels.size(1), gt_labels.size(2)),
                                    mode="bilinear", align_corners=True)

    probs = F.softmax(pred_logits, dim=1)
    target_onehot = F.one_hot(gt_labels, num_classes=_edgecwmi_state['num_classes']).permute(0,3,1,2).float()

    eps = 1e-7
    cwmi_val = torch.sum(target_onehot * torch.log((target_onehot + eps) / (probs + eps))) / (n*h*w)

    # --- 更新历史损失并调整权重 ---
    if epoch is not None:
        _edgecwmi_state['history_edge'].append(edge_val.item())
        _edgecwmi_state['history_cwmi'].append(cwmi_val.item())
        if len(_edgecwmi_state['history_edge']) > 2:
            _edgecwmi_state['history_edge'].pop(0)
            _edgecwmi_state['history_cwmi'].pop(0)

        if len(_edgecwmi_state['history_edge']) == 2:
            rate_edge = _edgecwmi_state['history_edge'][-1] / (_edgecwmi_state['history_edge'][-2] + eps)
            rate_cwmi = _edgecwmi_state['history_cwmi'][-1] / (_edgecwmi_state['history_cwmi'][-2] + eps)

            exp_edge = torch.exp(torch.tensor(rate_edge))
            exp_cwmi = torch.exp(torch.tensor(rate_cwmi))

            _edgecwmi_state['w_edge'] = (2 * exp_edge / (exp_edge + exp_cwmi)).item()
            _edgecwmi_state['w_cwmi'] = (2 * exp_cwmi / (exp_edge + exp_cwmi)).item()

    # --- 总损失 ---
    out = _edgecwmi_state['w_edge'] * edge_val + _edgecwmi_state['w_cwmi'] * cwmi_val
    return out
