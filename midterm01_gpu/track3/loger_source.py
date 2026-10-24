"""Optional LoGeR frontend: recompute a trailing prefix and emit only its last frame.

No future image is loaded, no old PLY/pose is replaced. Official PLY is the default.
This is a different reconstruction policy from the provided offline W32 script.
"""
from collections import deque
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from .io import sha256,write_json

RDF_TO_FLU=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
VENDOR=Path('/workspace/midterm01_gpu/vendor/LoGeR')


def load_model(device='cuda'):
    import torch
    import yaml
    sys.path.insert(0,str(VENDOR/'vendor/loger'))
    from loger.models.pi3 import Pi3
    config=yaml.safe_load((VENDOR/'vendor/loger/original_config.yaml').read_text())['model']
    config={k:v for k,v in config.items() if k in inspect.signature(Pi3.__init__).parameters}
    model=Pi3(**config)
    checkpoint=torch.load(VENDOR/'latest.pt',map_location='cpu',weights_only=True,mmap=True)
    state=checkpoint.get('model_state_dict',checkpoint)
    state={k.removeprefix('module.'):v for k,v in state.items()}
    model.load_state_dict(state,strict=True,assign=True)
    model=model.eval().to(device=device,dtype=torch.bfloat16 if device=='cuda' else torch.float32)
    # Vendor TTT explicitly disables autocast and passes x.float() to lr_fc.
    # Preserve that small projection in FP32 while reducing the large backbone.
    for module in model.modules():
        if hasattr(module,'lr_fc'): module.lr_fc.float()
    for name in ('point_head','conf_head','camera_head','metric_head'):
        head=getattr(model,name,None)
        if head is not None: head.float()
    return model


class CausalLoGeR:
    def __init__(self,data,run,context=2):
        import cv2
        import yaml
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('LoGeR requires CUDA; no CPU fallback')
        self.cv2=cv2;self.torch=torch;self.history=deque(maxlen=context)
        self.model=load_model();self.run=run;self.last_stamp=-1
        self.out=run/'reconstructed_ply';self.out.mkdir(exist_ok=False)
        camera=data/'Track3_images/camera_info'
        c=yaml.safe_load((camera/'gige_100_f_hdr_camera_info.yaml').read_text())
        K=np.array(c['camera_matrix']['data']).reshape(3,3)
        D=np.array(c['distortion_coefficients']['data'])
        P=np.array(c['projection_matrix']['data']).reshape(3,4)[:,:3]
        size=(c['image_width'],c['image_height'])
        self.maps=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),P,size,cv2.CV_32FC1)
        mask=cv2.imread(str(camera/'gige_100_f_hdr_mask.png'),0)
        if mask is None or mask.shape[::-1]!=size: raise ValueError('Invalid front ego mask')
        mask=cv2.remap(mask,*self.maps,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        self.excluded=cv2.resize(mask,(504,280),interpolation=cv2.INTER_NEAREST)>0
        write_json(run/'loger_frontend.json',dict(policy='trailing prefix; emit last frame only',context_frames=context,
                    coordinate_frame='camera-own FLU',confidence_min=0.18,checkpoint_sha256=sha256(VENDOR/'latest.pt'),
                    sources={str(p.relative_to(VENDOR)):sha256(p) for p in (VENDOR/'vendor').rglob('*.py')},
                    limitation='Causal reconstruction differs from teacher W32; no real inference/accuracy validation yet'))

    def __call__(self,path):
        import re
        cv2=self.cv2;torch=self.torch
        stamp=int(re.search(r'_ts_(\d+)',path.stem)[1])
        if stamp<=self.last_stamp: raise ValueError('LoGeR must receive frames once in timestamp order')
        image=cv2.imread(str(path))
        if image is None: raise ValueError(f'Cannot read {path}')
        image=cv2.remap(image,*self.maps,cv2.INTER_LINEAR)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        from PIL import Image
        rgb=np.asarray(Image.fromarray(image).resize((504,280),Image.Resampling.LANCZOS)).copy()
        self.history.append(rgb);self.last_stamp=stamp
        inputs=torch.from_numpy(np.stack(self.history)).permute(0,3,1,2).float().div(255).unsqueeze(0).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            pred=self.model(inputs,window_size=inputs.shape[1],overlap_size=0,se3=False,sim3=False)
        points=pred['local_points'][0,-1].float().cpu().numpy()
        confidence=torch.sigmoid(pred['conf'][0,-1]).float().cpu().numpy().squeeze()
        valid=np.isfinite(points).all(-1)&(points[...,2]>0)&(confidence>=0.18)&~self.excluded
        xyz=points[valid]@RDF_TO_FLU.T;colors=rgb[valid]
        if len(xyz)<100: raise ValueError('LoGeR returned insufficient valid points')
        from plyfile import PlyData,PlyElement
        array=np.empty(len(xyz),dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
        for i,k in enumerate(('x','y','z')):array[k]=xyz[:,i]
        for i,k in enumerate(('red','green','blue')):array[k]=colors[:,i]
        PlyData([PlyElement.describe(array,'vertex')],text=True).write(str(self.out/path.with_suffix('.ply').name))
        return xyz
