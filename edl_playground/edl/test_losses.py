import unittest
import torch
import math

from torch.testing import assert_close

from losses import (
    _type_2_maximum_likelihood_loss,
    _bayes_risk_for_cross_entropy_loss,
    _bayes_risk_for_sse_loss,
    _KL_divergence,
    _edl_loss,
    EDL_Loss,
    get_belief_probs_and_uncertainty,
    get_correct_preds,
)
class Test_EDL_Loss(unittest.TestCase):

    def test_type_2_maximum_likelihood(self):
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        S = torch.sum(alpha, dim=1)

        likelihood = _type_2_maximum_likelihood_loss(y, alpha, S)

        target = torch.tensor([
            math.log(42) - math.log(1),
            math.log(5) - math.log(2)
        ])

        self.assertEqual(likelihood.shape, torch.Size([y.shape[0]]))
        assert_close(likelihood, target)

    def test_bayes_risk_for_cross_entropy_loss(self):
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        S = torch.sum(alpha, dim=1)

        likelihood = _bayes_risk_for_cross_entropy_loss(y, alpha, S)

        target = torch.tensor([
            torch.digamma(torch.tensor(42)).item() - torch.digamma(torch.tensor(1)).item(),
            torch.digamma(torch.tensor(5)).item() - torch.digamma(torch.tensor(2)).item()
        ])

        self.assertEqual(likelihood.shape, torch.Size([y.shape[0]]))
        assert_close(likelihood, target)

    def test_bayes_risk_for_sse_loss(self):
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        S = torch.sum(alpha, dim=1)

        likelihood = _bayes_risk_for_sse_loss(y, alpha, S)

        target = torch.tensor([
            (41 / 42)**2 + 41 * (42-41) / (42**2 * (42+1)) + (1 - 1/42)**2 + 1 * (42-1) / (42**2 * (42+1)),
            (1- 2/5)**2 + 2*(5-2) / (5**2 * (5+1)) + (3/5)**2 + 3*(5-3) / (5**2 * (5+1))
        ])

        self.assertEqual(likelihood.shape, torch.Size([y.shape[0]]))
        assert_close(likelihood, target)

    def test_KL_divergence(self):
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])

        alpha_tilde = y + (1 - y) * alpha

        divergence = _KL_divergence(alpha_tilde)

        target = torch.tensor([
            math.log(math.gamma(42) / (math.gamma(2) * math.gamma(41) * math.gamma(1))) \
                + 40*(torch.digamma(torch.tensor(41)).item() - torch.digamma(torch.tensor(42)).item()),
            math.log(math.gamma(4) / (math.gamma(2) * math.gamma(1) * math.gamma(3))) \
                + 2*(torch.digamma(torch.tensor(3)).item() - torch.digamma(torch.tensor(4)).item()),
        ])

        self.assertEqual(divergence.shape, torch.Size([y.shape[0]]))
        assert_close(divergence, target)

    def test_edl_loss(self):
        self.test_KL_divergence()
        self.test_type_2_maximum_likelihood()
        self.test_bayes_risk_for_cross_entropy_loss()
        self.test_bayes_risk_for_sse_loss()

        loss_func = _type_2_maximum_likelihood_loss
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        S = torch.sum(alpha, dim=1)

        alpha_tilde = y + (1 - y) * alpha

        divergence = _KL_divergence(alpha_tilde)

        loss = _type_2_maximum_likelihood_loss(y, alpha, S)
        
        training_epoch = 42

        annealing_coeff = min(1, training_epoch/10)

        loss_target = loss + annealing_coeff * divergence

        loss_None = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, None)
        loss_none = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, "none")
        loss_mean = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, "mean")
        loss_sum = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, "sum")

        assert_close(loss_None, loss_target)
        assert_close(loss_none, loss_target)
        assert_close(loss_mean, (loss_target[0] + loss_target[1]) / 2)
        assert_close(loss_sum, loss_target[0] + loss_target[1])

        ##################################################################################
        
        loss_func = _bayes_risk_for_cross_entropy_loss
        
        loss = _bayes_risk_for_cross_entropy_loss(y, alpha, S)

        loss_target = loss + annealing_coeff * divergence

        loss_None = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, None)

        assert_close(loss_None, loss_target)

        ##################################################################################
        
        loss_func = _bayes_risk_for_sse_loss
        
        loss = _bayes_risk_for_sse_loss(y, alpha, S)

        loss_target = loss + annealing_coeff * divergence

        loss_None = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, None)

        assert_close(loss_None, loss_target)

        ##################################################################################

        training_epoch = 1

        annealing_coeff = min(1, training_epoch/10)
        
        loss_func = _bayes_risk_for_sse_loss
        
        loss = _bayes_risk_for_sse_loss(y, alpha, S)

        loss_target = loss + annealing_coeff * divergence

        loss_None = _edl_loss(loss_func, training_epoch, y, alpha, S, alpha_tilde, None)

        assert_close(loss_None, loss_target)

    def test_EDL_Loss(self):
        self.test_edl_loss()

        alpha = torch.tensor([[41, 1], [2, 3]])
        evidence = alpha - 1
        target = torch.tensor([1, 0])

        training_epoch = 1
        reduction = "mean"

        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        S = torch.sum(alpha, dim=1)
        alpha_tilde = y + (1 - y) * alpha

        loss_fn = EDL_Loss("ml")

        loss_target = _edl_loss(_type_2_maximum_likelihood_loss, training_epoch, y, alpha, S, alpha_tilde, reduction)

        loss = loss_fn(evidence, target, training_epoch, reduction)

        assert_close(loss, loss_target)

        ##################################################################

        loss_fn = EDL_Loss("ce")

        loss_target = _edl_loss(_bayes_risk_for_cross_entropy_loss, training_epoch, y, alpha, S, alpha_tilde, reduction)

        loss = loss_fn(evidence, target, training_epoch, reduction)

        assert_close(loss, loss_target)

        ##################################################################

        loss_fn = EDL_Loss("sse")

        loss_target = _edl_loss(_bayes_risk_for_sse_loss, training_epoch, y, alpha, S, alpha_tilde, reduction)

        loss = loss_fn(evidence, target, training_epoch, reduction)

        assert_close(loss, loss_target)

    def test_get_belief_probs_and_uncertainty(self):
        y = torch.tensor([[0, 1], [1, 0]])
        alpha = torch.tensor([[41, 1], [2, 3]])
        K = 2

        evidence = alpha - 1

        belief, probs, uncertainty = get_belief_probs_and_uncertainty(evidence, K)

        belief_target = torch.tensor([
            [40/42, 0],
            [1/5, 2/5]
        ])

        probs_target = torch.tensor([
            [41/42, 1/42],
            [2/5, 3/5]
        ])

        uncertainty_target = torch.tensor([
            2/42,
            2/5
        ])

        self.assertEqual(belief.shape, torch.Size(y.shape))
        self.assertEqual(probs.shape, torch.Size(y.shape))
        self.assertEqual(uncertainty.shape, torch.Size([y.shape[0]]))
        assert_close(belief, belief_target)
        assert_close(probs, probs_target)
        assert_close(uncertainty, uncertainty_target)

    def test_get_correct_preds(self):
        self.test_get_belief_probs_and_uncertainty()
        alpha = torch.tensor([[41, 1], [2, 3]])

        evidence = alpha - 1

        probs = torch.tensor([
            [41/42, 1/42],
            [2/5, 3/5]
        ])

        uncertainty = torch.tensor([
            2/42,
            2/5
        ])

        #####################################################

        target = torch.tensor([1, 0])
        uncertainty_thresh = 0.6

        correct_target = 0
        rejected_corrects_target = 0

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        target = torch.tensor([1, 1])
        uncertainty_thresh = 1/5

        correct_target = 0
        rejected_corrects_target = 1

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        uncertainty_thresh = 2/5

        correct_target = 1
        rejected_corrects_target = 0

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        uncertainty_thresh = 3/5

        correct_target = 1
        rejected_corrects_target = 0

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        target = torch.tensor([0, 1])
        uncertainty_thresh = 1/42

        correct_target = 0
        rejected_corrects_target = 2

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        uncertainty_thresh = 2/42

        correct_target = 1
        rejected_corrects_target = 1

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        uncertainty_thresh = 1/5

        correct_target = 1
        rejected_corrects_target = 1

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        uncertainty_thresh = 2/5

        correct_target = 2
        rejected_corrects_target = 0

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)

        #####################################################

        uncertainty_thresh = 3/5

        correct_target = 2
        rejected_corrects_target = 0

        correct, rejected_corrects = get_correct_preds(evidence, target, uncertainty_thresh)

        self.assertEqual(correct, correct_target)
        self.assertEqual(rejected_corrects, rejected_corrects_target)


if __name__ == '__main__':
    unittest.main()