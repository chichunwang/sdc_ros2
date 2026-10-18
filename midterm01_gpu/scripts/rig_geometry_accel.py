"""與既有取樣規則相同的批次實作；CUDA PCA／整數排序，CPU exact KD-tree。"""
from pathlib import Path
import time
import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree


class CachedRectifier:
    """Cache only calibration-derived maps/mask. Same OpenCV remap and decoded pixels."""
    def __init__(self):
        self.maps = {}

    def __call__(self, image_path, calibration_path, size=(448,294), exclusion_mask=None):
        key = (str(Path(calibration_path).resolve()),tuple(size),str(Path(exclusion_mask).resolve()) if exclusion_mask else None)
        if key not in self.maps:
            info = yaml.safe_load(Path(calibration_path).read_text())
            if info['distortion_model'] != 'fisheye':raise ValueError('Only calibrated fisheye supported')
            k=np.array(info['camera_matrix']['data']).reshape(3,3)
            d=np.array(info['distortion_coefficients']['data'])
            r=np.array(info['rectification_matrix']['data']).reshape(3,3)
            np.testing.assert_allclose(r,np.eye(3),atol=1e-8)
            new_k=np.array(info['projection_matrix']['data']).reshape(3,4)[:,:3].copy()
            new_k[0]*=size[0]/info['image_width'];new_k[1]*=size[1]/info['image_height']
            mx,my=cv2.fisheye.initUndistortRectifyMap(k,d,r,new_k,size,cv2.CV_32FC1)
            valid=(mx>=1)&(mx<info['image_width']-2)&(my>=1)&(my<info['image_height']-2)
            if exclusion_mask:
                excluded=cv2.imread(str(exclusion_mask),cv2.IMREAD_GRAYSCALE)
                if excluded is None or excluded.shape!=(info['image_height'],info['image_width']):raise ValueError('Exclusion mask size mismatch')
                valid &= cv2.remap(excluded,mx,my,cv2.INTER_NEAREST,borderValue=255)==0
            self.maps[key]=(mx,my,new_k.astype(np.float32),valid,(info['image_height'],info['image_width']))
        mx,my,k,valid,shape=self.maps[key]
        image=cv2.imread(str(image_path))
        if image is None or image.shape[:2]!=shape:raise ValueError('Raw image/calibration size mismatch')
        rgb=cv2.cvtColor(cv2.remap(image,mx,my,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        return rgb,k.copy(),valid.copy()


def select_cells(points, camera_ids, confidence, candidates, accepted, axes, budget):
    """Vectorized exact camera/axis + cell round-robin; confidence ties use original index."""
    original=candidates[accepted]
    if not len(original):return np.empty(0,dtype=int),{'eligible':len(candidates),'quota_per_nonempty_bin':budget,'bins':{}}
    group=camera_ids[original].astype(np.int64)*3+axes[accepted]
    cells=np.floor(points[original]/2.).astype(np.int64)
    order=np.lexsort((original,-confidence[original],cells[:,2],cells[:,1],cells[:,0],group))
    original,group,cells=original[order],group[order],cells[order]
    starts=np.r_[True,(group[1:]!=group[:-1])|np.any(cells[1:]!=cells[:-1],axis=1)]
    depth=np.arange(len(original))-np.maximum.accumulate(np.where(starts,np.arange(len(original)),0))
    cell_groups=group.copy()
    sequence=np.lexsort((cells[:,2],cells[:,1],cells[:,0],depth,group))
    original,group=original[sequence],group[sequence]
    groups,available=np.unique(group,return_counts=True)
    quota=budget//len(groups)
    begin=np.r_[0,np.cumsum(available)[:-1]]
    counts={};chosen=[]
    for g,n,start in zip(groups,available,begin):
        take=min(int(n),quota);chosen.append(original[start:start+take])
        counts[f'camera{g//3}_normal{g%3}']={'available':int(n),'selected':take,
                                         'cells':int(np.count_nonzero(starts & (cell_groups==g)))}
    return np.sort(np.concatenate(chosen)),{'eligible':len(candidates),'quota_per_nonempty_bin':quota,'bins':counts}


class GeometryAccelerator:
    def __init__(self,backend='cuda',chunk_size=256):
        if backend not in ('cpu','cuda'):raise ValueError('Explicit cpu or cuda backend required')
        self.backend,self.chunk_size=backend,chunk_size
        self.stats={}
        self.torch=None
        if backend=='cuda':
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA geometry requested but unavailable; no CPU fallback')
            self.torch=torch

    def voxel_indices(self,points,leaf=.3):
        # Keep CPU float64 floor exactly. GPU operates on integers only.
        coords=np.floor(points/leaf).astype(np.int64)
        if self.backend=='cpu':
            _,keep=np.unique(coords,axis=0,return_index=True);return np.sort(keep)
        torch=self.torch
        if not len(coords):return np.empty(0,dtype=int)
        low=coords.min(axis=0);span=coords.max(axis=0)-low+1
        if int(span[0])*int(span[1])*int(span[2])>=2**63:raise ValueError('Voxel key exceeds exact int64 packing')
        keys=((coords[:,0]-low[0])*int(span[1])+(coords[:,1]-low[1]))*int(span[2])+(coords[:,2]-low[2])
        tensor=torch.from_numpy(keys).to('cuda')
        values,order=torch.sort(tensor,stable=True)
        first=torch.cat((torch.ones(1,dtype=torch.bool,device='cuda'),values[1:]!=values[:-1]))
        return np.sort(order[first].cpu().numpy())

    def classify(self,p,distances,neighbors):
        accepted=np.zeros(len(p),dtype=bool);axes=np.zeros(len(p),dtype=np.int64)
        refinements=0
        for start in range(0,len(p),self.chunk_size):
            end=min(start+self.chunk_size,len(p));mask=distances[start:end]<=1.5
            count=mask.sum(axis=1)
            local=p[neighbors[start:end]]
            if self.backend=='cuda':
                torch=self.torch
                a=torch.from_numpy(local).to('cuda');m=torch.from_numpy(mask).to('cuda')
                n=torch.from_numpy(np.maximum(count,1)).to('cuda')
                mean=(a*m[...,None]).sum(dim=1)/n[:,None]
                centered=(a-mean[:,None,:])*m[...,None]
                covariance=centered.transpose(1,2)@centered/n[:,None,None]
                values,vectors=torch.linalg.eigh(covariance,UPLO='L')
                values,vectors=values.cpu().numpy(),vectors.cpu().numpy()
            else:
                mean=np.sum(local*mask[...,None],axis=1)/np.maximum(count,1)[:,None]
                centered=(local-mean[:,None,:])*mask[...,None]
                covariance=np.swapaxes(centered,1,2)@centered/np.maximum(count,1)[:,None,None]
                values,vectors=np.linalg.eigh(covariance)
            with np.errstate(divide='ignore',invalid='ignore'):
                quality=(values[:,1]-values[:,0])/values[:,2]
            normal=np.abs(vectors[:,:,0]);axis=normal.argmax(axis=1)
            good=(count>=6)&(values[:,1]>=.01)&(values[:,2]>1e-8)&(quality>=.2)&(values[:,0]<=.25*values[:,1])
            sorted_normal=np.sort(normal,axis=1)
            # Resolve near-threshold/near-tie arithmetic with the original scalar CPU calculation.
            tol=1e-8
            uncertain=(count>=6)&((np.abs(values[:,1]-.01)<=tol)|((values[:,1]>=.01-tol)&
                      ((np.abs(values[:,2]-1e-8)<=tol)|(np.abs(quality-.2)<=tol)|
                       (np.abs(values[:,0]-.25*values[:,1])<=tol)|(good&(sorted_normal[:,2]-sorted_normal[:,1]<=tol)))))
            for j in np.flatnonzero(uncertain):
                patch=local[j,mask[j]];patch=patch-patch.mean(axis=0)
                vv,ee=np.linalg.eigh(patch.T@patch/len(patch))
                good[j]=(vv[1]>=.01 and vv[2]>1e-8 and (vv[1]-vv[0])/vv[2]>=.2 and vv[0]<=.25*vv[1])
                axis[j]=int(np.argmax(np.abs(ee[:,0])))
            refinements+=int(uncertain.sum());accepted[start:end]=good;axes[start:end]=axis
        self.stats['scalar_boundary_refinements']=refinements
        return accepted,axes

    def balanced_indices(self,points,camera_ids,confidence,budget=6000,leaf=.3):
        self.stats={};started=time.perf_counter()
        points,camera_ids,confidence=np.asarray(points),np.asarray(camera_ids),np.asarray(confidence)
        if len(points)!=len(camera_ids) or len(points)!=len(confidence) or budget<1:raise ValueError('Invalid sampling input')
        if not np.all(np.isfinite(points)) or not np.all(np.isfinite(confidence)):raise ValueError('Nonfinite sampling input')
        candidates=np.concatenate([np.flatnonzero(camera_ids==cam)[self.voxel_indices(points[camera_ids==cam],leaf)] for cam in np.unique(camera_ids)])
        self.stats['voxel_seconds']=time.perf_counter()-started
        if len(candidates)<16:return candidates[:budget],{'eligible':len(candidates),'bins':{}}
        p=points[candidates];t=time.perf_counter();distances,neighbors=cKDTree(p).query(p,k=16)
        self.stats['exact_cpu_knn_seconds']=time.perf_counter()-t
        t=time.perf_counter();accepted,axes=self.classify(p,distances,neighbors)
        self.stats['pca_seconds']=time.perf_counter()-t
        t=time.perf_counter();selected,report=select_cells(points,camera_ids,confidence,candidates,accepted,axes,budget)
        self.stats['cell_selection_seconds']=time.perf_counter()-t
        self.stats['total_seconds']=time.perf_counter()-started
        return selected,report
