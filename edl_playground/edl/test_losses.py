import unittest
import torch
import math

from torch.testing import assert_close

from .losses import (
    _type_2_maximum_likelihood_loss,
    _bayes_risk_for_cross_entropy_loss,
    _bayes_risk_for_sse_loss
)


class Test_EDL_SSE_Loss(unittest.TestCase):

    def test_type_2_maximum_likelihood(self):
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        S = torch.sum(alpha, dim=1)
        print(S)
        likelihood = _type_2_maximum_likelihood_loss(y, alpha, S)
        print(likelihood)

        target = torch.tensor([
            math.log(42) - math.log(1),
            math.log(5) - math.log(2)
        ])
        print(target)

        self.assertEqual(likelihood.shape, torch.Size([y.shape[0]]))
        assert_close(likelihood, target)

if __name__ == '__main__':
    unittest.main()