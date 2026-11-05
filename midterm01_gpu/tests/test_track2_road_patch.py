"""Road-depth assumptions require actual support of the sparse LK patch."""
import cv2
import numpy as np
import pytest
from track2 import parameters as p
from track2.image_speed import CausalGroundSpeed, LK_WINDOW, road_patch_mask


@pytest.mark.parametrize('shape,region', [((45,55),(0,45,0,55)), ((65,80),(10,55,15,65)),
                                        ((45,55),(20,25,0,55))])
def test_patch_support_matches_every_pixel_in_window(shape,region):
    mask=np.zeros(shape,np.uint8);y0,y1,x0,x1=region;mask[y0:y1,x0:x1]=255
    before=mask.copy();supported=road_patch_mask(mask)
    ry,rx=np.array(LK_WINDOW)//2
    for y in range(shape[0]):
        for x in range(shape[1]):
            valid=(ry<=y<shape[0]-ry and rx<=x<shape[1]-rx
                   and bool(np.all(mask[y-ry:y+ry+1,x-rx:x+rx+1])))
            assert bool(supported[y,x]) == valid
    np.testing.assert_array_equal(mask,before)


def test_narrow_mask_cannot_create_road_velocity_but_disabled_branch_is_exact(monkeypatch):
    monkeypatch.setattr(p,'FUSION_PROFILE','v52_directional_memory')
    monkeypatch.setattr(p,'V510_ENABLED',True);monkeypatch.setattr(p,'V5162_ENABLED',True)
    for name in ['V511_ENABLED','V512_ENABLED','V5121_ENABLED','V513_ENABLED','V514_ENABLED','V515_ENABLED']:
        monkeypatch.setattr(p,name,False)
    import track2.image_speed as module
    tool=CausalGroundSpeed.__new__(CausalGroundSpeed)
    mask=np.zeros((100,160),np.uint8);mask[60:69,5:155]=255
    gray=np.random.default_rng(5163).integers(0,256,mask.shape,dtype=np.uint8)
    tool.anchor=dict(gray=gray,ground=dict(mask=mask,normal=[0.,0.,1.],height=1.5,height_variance=.01))
    tool.K=np.array([[100.,0.,80.],[0.,100.,50.],[0.,0.,1.]])
    tool.gyro_angle=0.
    original_track=module.track_images;seen=[]
    def observe_tracks(before,after,actual_mask,**kwargs):
        seen.append(actual_mask.copy())
        return original_track(before,after,actual_mask,**kwargs)
    monkeypatch.setattr(module,'track_images',observe_tracks)
    monkeypatch.setattr(p,'V5163_ENABLED',False)
    control=tool._ground_observation(gray,{},.25)
    np.testing.assert_array_equal(seen[-1],mask)
    assert 'road_patch_support' not in control
    monkeypatch.setattr(p,'V5163_ENABLED',True)
    revised=tool._ground_observation(gray,{},.25)
    assert not seen[-1].any()
    assert revised['ground_tracks']==0 and not revised['accepted']
    assert revised['reason']=='insufficient_ground_tracks'
    assert revised['road_patch_support']['window_px']==list(LK_WINDOW)
    assert revised['road_patch_support']['supported_mask_pixels']==0
    monkeypatch.setattr(p,'V5163_ENABLED',False)
    assert tool._ground_observation(gray,{},.25)==control


def test_guard_follows_causal_turn_profile(monkeypatch):
    monkeypatch.setattr(p,'V5163_ENABLED',True)
    monkeypatch.setattr(p,'V5162_ENABLED',False)
    assert not p.v5163_enabled()
