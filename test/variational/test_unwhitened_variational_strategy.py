#!/usr/bin/env python3

import unittest

import torch

import gpytorch
from gpytorch.test.variational_test_case import VariationalTestCase


class TestUnwhitenedVariationalGP(VariationalTestCase, unittest.TestCase):
    @property
    def batch_shape(self):
        return torch.Size([])

    @property
    def distribution_cls(self):
        return gpytorch.variational.CholeskyVariationalDistribution

    @property
    def mll_cls(self):
        return gpytorch.mlls.VariationalELBO

    @property
    def strategy_cls(self):
        return gpytorch.variational.UnwhitenedVariationalStrategy

    def test_training_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_training_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        if self.distribution_cls == gpytorch.variational.CholeskyVariationalDistribution:
            self.assertEqual(cholesky_mock.call_count, 3)  # One for each forward pass, once for initialization
        else:
            self.assertEqual(cholesky_mock.call_count, 2)  # One for each forward pass

    def test_eval_iteration(self, *args, **kwargs):
        cg_mock, cholesky_mock, ciq_mock = super().test_eval_iteration(*args, **kwargs)
        self.assertFalse(cg_mock.called)
        self.assertFalse(ciq_mock.called)
        self.assertEqual(cholesky_mock.call_count, 1)  # One to compute cache, that's it!

    def test_fantasy_call(self, *args, **kwargs):
        # we only want to check CholeskyVariationalDistribution
        if self.distribution_cls is gpytorch.variational.CholeskyVariationalDistribution:
            return super().test_fantasy_call(*args, **kwargs)

        with self.assertRaises(NotImplementedError):
            super().test_fantasy_call(*args, **kwargs)

    def test_eval_mode_allows_repeated_backward(self, *args, **kwargs):
        """Repeated eval-mode backward passes should rebuild cached Cholesky factors."""
        model, _ = self._make_model_and_likelihood(
            num_inducing=5,
            batch_shape=self.batch_shape,
            strategy_cls=self.strategy_cls,
            distribution_cls=self.distribution_cls,
        )
        model.eval()

        cached_cholesky_factors = []
        for _ in range(2):
            test_x = torch.randn(*self.batch_shape, 3, 2, requires_grad=True)
            test_y = torch.randn(*self.batch_shape, 3)

            predictive_dist = model(test_x)
            cached_cholesky = model.variational_strategy._memoize_cache["cholesky_factor"]
            cached_cholesky_factors.append(cached_cholesky)

            loss = (predictive_dist.mean - test_y).mean()
            loss.backward()
            self.assertNotIn("cholesky_factor", model.variational_strategy._memoize_cache)
            self.assertIsNotNone(test_x.grad)

        self.assertIsNot(*cached_cholesky_factors)


class TestUnwhitenedPredictiveGP(TestUnwhitenedVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.PredictiveLogLikelihood


class TestUnwhitenedRobustVGP(TestUnwhitenedVariationalGP):
    @property
    def mll_cls(self):
        return gpytorch.mlls.GammaRobustVariationalELBO


class TestUnwhitenedMeanFieldVariationalGP(TestUnwhitenedVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldPredictiveGP(TestUnwhitenedPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedMeanFieldRobustVGP(TestUnwhitenedRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.MeanFieldVariationalDistribution


class TestUnwhitenedDeltaVariationalGP(TestUnwhitenedVariationalGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaPredictiveGP(TestUnwhitenedPredictiveGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


class TestUnwhitenedDeltaRobustVGP(TestUnwhitenedRobustVGP):
    @property
    def distribution_cls(self):
        return gpytorch.variational.DeltaVariationalDistribution


if __name__ == "__main__":
    unittest.main()
