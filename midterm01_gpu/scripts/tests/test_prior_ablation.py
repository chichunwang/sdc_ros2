import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prior_ablation import predict_positions, select_windows


class PriorAblationTests(unittest.TestCase):
    def test_held_out_values_do_not_change_either_prediction(self):
        stamps=1727234445000000000+np.arange(120,dtype=np.int64)*250000000
        t=(stamps-stamps[0])/1e9
        xy=np.column_stack((10*t,3*t+.02*t*t))
        training=np.ones(len(t),dtype=bool);training[48:54]=False
        a,b,meta=predict_positions(stamps,xy,stamps[49:53],training)
        changed=xy.copy();changed[~training]+=[1000,-1000]
        aa,bb,_=predict_positions(stamps,changed,stamps[49:53],training)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        self.assertEqual(meta['position_measurement_updates'],114)

    def test_integer_epoch_translation_preserves_predictions(self):
        stamps=np.arange(100,dtype=np.int64)*250000000
        xy=np.column_stack((stamps/1e8,stamps/2e8))
        train=np.ones(100,dtype=bool);train[40:45]=False
        first=predict_positions(stamps,xy,stamps[40:45],train)
        offset=np.int64(1727234445000000000)
        second=predict_positions(stamps+offset,xy,stamps[40:45]+offset,train)
        for a,b in zip(first[:2],second[:2]):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(first[0],xy[40:45],atol=1e-10,rtol=0)

    def test_window_selection_covers_motion_classes_without_overlap(self):
        t=np.arange(900)*.04
        speed=np.r_[np.zeros(300),np.full(600,10.)]
        xy=np.column_stack((np.cumsum(speed)*.04,np.zeros(900)))
        heading=np.r_[np.zeros(600),np.arange(300)*np.deg2rad(5)*.04]
        windows,missing=select_windows(np.int64(1727234445000000000)+(t*1e9).astype(np.int64),xy,heading)
        self.assertFalse(missing)
        self.assertEqual({w['class'] for w in windows},{'low_speed','straight','turning'})
        self.assertTrue(all(abs(a['start']-b['start'])>=48 for i,a in enumerate(windows) for b in windows[i+1:]))


if __name__=='__main__':unittest.main()
