import project_path
import numpy as np
from src.util.t2m import t2m, convert_index 
from src.util.m2t import m2t  
from src.util.soft_treshold import soft_treshold
import unittest


class TestTensor(unittest.TestCase):

    def test_convert_index(self):
        n = (3,2,2)
        i1 = (0,0,0); i2=(1,1,0); i3=(1,0,1); i4=(2,1,1); 
        k=1 # Test first mode unfoldings
        self.assertEqual(convert_index(n,i1,k),(0,0))
        self.assertEqual(convert_index(n,i2,k),(1,2))
        self.assertEqual(convert_index(n,i3,k),(1,1))
        self.assertEqual(convert_index(n,i4,k),(2,3))
        
        k=2 # Test second mode unfoldings
        self.assertEqual(convert_index(n,i1,k),(0,0))
        self.assertEqual(convert_index(n,i2,k),(1,1))
        self.assertEqual(convert_index(n,i3,k),(0,4))
        self.assertEqual(convert_index(n,i4,k),(1,5))
        
        k=3 # Test third mode unfoldings
        self.assertEqual(convert_index(n,i1,k),(0,0))
        self.assertEqual(convert_index(n,i2,k),(0,3))
        self.assertEqual(convert_index(n,i3,k),(1,2))
        self.assertEqual(convert_index(n,i4,k),(1,5))

        n = (2,3,3,2); i1 = (1,2,0,1)
        self.assertEqual(convert_index(n,i1,1),(1,13))
        self.assertEqual(convert_index(n,i1,2),(2,3))
        self.assertEqual(convert_index(n,i1,3),(0,11))
        self.assertEqual(convert_index(n,i1,4),(1,15))

    def test_t2m(self):
        """Test unfolding operation in each mode.
        Generate a random tensor, apply unfolding operation on each mode and test if the mapping is done correctly for 10 samples.
        """
        n = [2,3,4,5]
        rng = np.random.default_rng()
        X = rng.random(n)
        Xunfolded = [t2m(X,k+1) for k in range(len(n))]
        # Test if t2m returns matrices with correct shapes
        for k in range(len(n)):
            self.assertEqual(Xunfolded[k].shape, (n[k],np.prod(n)//n[k]))
        
        # Test 10 random elements if they are correctly mapped.
        for _ in range(10):
            i = tuple([rng.integers(0,en) for en in n])
            for k in range(len(n)):
                Xk = Xunfolded[k]
                i_ = convert_index(tuple(n),i,k+1)
                self.assertEqual(X[i], Xk[i_])

    def test_m2t(self):
        """Test folding operation for each mode.
        Create a random tensor, unfold it using t2m in every mode and fold them back to check if the operation preserves the correct order.
        """
        n = (4,5,3,2)
        rng = np.random.default_rng()
        X = rng.random(n)
        for k in range(len(n)):
            X_ = t2m(X, k+1)
            self.assertTrue(np.alltrue( m2t(X_,n,k+1) == X) )


    def test_soft_threshold(self):
        """Test soft thresholding function for vector and matrix inputs."""
        # Test vector case
        x = np.array([-3,-2,-1,0,1,2,3])
        
        self.assertTrue(np.alltrue(
            soft_treshold(x,1.5)==np.array([-1.5,-0.5,0,0,0,0.5,1.5])
        ))
        # Test Matrix Case
        X = np.arange(-3,5).reshape(2,4)
        self.assertTrue(np.alltrue(
            soft_treshold(X,2)==np.array([[-1,0,0,0],[0,0,1,2]])
        ))
        
        self.assertRaises(ValueError, soft_treshold, X,-2)

    def t2m_multi(self):
        pass