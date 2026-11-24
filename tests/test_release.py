import copy
import unittest
import numpy as np
import torch
from disaster_response.datasets import checked_splits
from disaster_response.backbone_online_adaptation import run_backbone_stream


class ReleaseTests(unittest.TestCase):
    def test_target_purge_keeps_test(self):
        raw=dict(train=np.arange(30),val=np.arange(30,50),test=np.arange(50,70))
        splits,overlap=checked_splits(raw,90,'purged')
        self.assertEqual([len(splits[k]) for k in ('train','val','test')],[25,15,20])
        self.assertFalse(any(overlap.values()))
        np.testing.assert_array_equal(splits['test'],raw['test'])
        with self.assertRaises(ValueError):checked_splits(dict(train=np.arange(30),val=np.arange(20,50),test=np.arange(50,70)),90,'purged')

    def test_future_labels_do_not_change_earlier_forecasts(self):
        torch.set_num_threads(1);torch.manual_seed(0)
        n=2;count=16;width=4
        cache=dict(features=torch.randn(count,n,width),raw=torch.ones(count,6,n),
                   base=torch.full((count,6,n),12.),factor=torch.ones(count,6,n),original=torch.full((count,6,n),12.))
        flow=np.full((40,n),10.,np.float32);changed=flow.copy();changed[15:]=100.
        cfg=dict(mode='shared_readout',window=12,lr=.001,anchor=.1)
        a,audit,_,_=run_backbone_stream(cache,np.arange(count),lambda t:flow[t],[10.,10.],10.,2.,cfg,'cpu')
        b,_,_,_=run_backbone_stream(cache,np.arange(count),lambda t:changed[t],[10.,10.],10.,2.,cfg,'cpu')
        np.testing.assert_array_equal(a[:10],b[:10])
        self.assertGreater(float(np.abs(a[10:]-b[10:]).max()),0.)
        self.assertTrue(all(r['latest_feedback_hour'] is None or r['latest_feedback_hour']<r['first_target_hour'] for r in audit))


if __name__=='__main__':unittest.main()
