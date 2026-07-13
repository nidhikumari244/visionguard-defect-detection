#!/usr/bin/env python3
# VisionGuard Inference Script
# python inference.py --image img.jpg --category bottle

import torch, numpy as np, argparse
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import timm
from torchvision import transforms

THRESHOLDS = {'bottle':55.59,'cable':48.27,'carpet':124.20,'wood':87.90,'screw':66.95}

def predict(image_path, category, mb_dir='./memory_banks'):
    backbone = timm.create_model('wide_resnet50_2', pretrained=True,
        features_only=True, out_indices=(1,2))
    backbone.eval()
    T = transforms.Compose([
        transforms.Resize((256,256)), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    mb   = np.load(f'{mb_dir}/{category}_mb_sub_norm.npy')
    mean = np.load(f'{mb_dir}/{category}_mean.npy')
    std  = np.load(f'{mb_dir}/{category}_std.npy')
    nn   = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(mb)
    img  = Image.open(image_path).convert('RGB')
    t    = T(img).unsqueeze(0)
    with torch.no_grad(): f = backbone(t)
    f0,f1 = f[0],f[1]
    f1u = torch.nn.functional.interpolate(f1,size=f0.shape[2:],
        mode='bilinear',align_corners=False)
    c = torch.cat([f0,f1u],dim=1)
    b,ch,h,w = c.shape
    p = c.permute(0,2,3,1).reshape(-1,ch).numpy().astype(np.float32)
    d,_ = nn.kneighbors((p-mean)/std)
    score = float(d.max())
    thresh = THRESHOLDS[category]
    decision = 'ANOMALY DETECTED' if score>thresh else 'NORMAL'
    print(f'Decision: {decision} | Score: {score:.4f} | Threshold: {thresh}')
    return decision, score

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--category', required=True,
        choices=['bottle','cable','carpet','wood','screw'])
    p.add_argument('--memory_bank_dir', default='./memory_banks')
    a = p.parse_args()
    predict(a.image, a.category, a.memory_bank_dir)
