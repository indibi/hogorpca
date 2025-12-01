import project_path
import numpy as np

from src.util.qmult import qmult
from src.util.t2m import t2m
from src.util.generate_lr_data import generate_low_rank_data
import unittest


class TestSyntheticData(unittest.TestCase):

    def test_qmult(self):
        """Test if qmult produces an orthonormal matrix with asked shape."""
        for _ in range(3):
            n = np.random.randint(1,10)
            Q = qmult(n)
            self.assertEqual(Q.shape, (n,n))
            self.assertAlmostEqual(0, np.linalg.norm(np.eye(n)-Q@Q.T))

    def test_generate_lr_data(self):
        """Test if the generated low-rank data is truly low-rank in tucker rank sense"""
        
        dim = (10,10,10); ranks=(5,6,7)
        X = generate_low_rank_data(dim,ranks)
        for m in range(len(dim)):
            Xm  = t2m(X,m+1)
            U,S,V = np.linalg.svd(Xm,full_matrices=False)
            self.assertAlmostEqual(0, S[ranks[m]])

    
