"""Unit tests for bayes_classifier.GaussianBayes.  Run:  python -m unittest discover -s tests -v"""
import os, sys, unittest
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bayes_classifier import GaussianBayes, confusion


def toy(n_per_class=(400, 300, 200), seed=0, d=4):
    rng = np.random.default_rng(seed)
    means = [np.zeros(d), np.full(d, 2.0), np.array([3.0, -2.0, 1.0, 0.0])[:d]]
    X, y = [], []
    for k, n in enumerate(n_per_class):
        A = rng.normal(size=(d, d)) * 0.5 + np.eye(d)
        X.append(rng.normal(size=(n, d)) @ A.T + means[k]); y += [k] * n
    return np.vstack(X), np.array(y)


class TestEstimation(unittest.TestCase):
    def test_estimates_recover_true_parameters(self):
        rng = np.random.default_rng(1)
        mu = np.array([1.0, -1.0, 0.5]); S = np.array([[1.0, .6, .2], [.6, 2.0, -.3], [.2, -.3, .8]])
        X = np.vstack([rng.multivariate_normal(mu, S, 40000), rng.multivariate_normal(-mu, S * .5, 40000)])
        y = np.repeat([0, 1], 40000)
        m = GaussianBayes(reg=0.0).fit(X, y)
        self.assertTrue(np.allclose(m.means_[0], mu, atol=0.05))
        self.assertTrue(np.allclose(m.covs_[0], S, atol=0.08))
        self.assertTrue(np.allclose(m.covs_[1], S * .5, atol=0.05))

    def test_priors_follow_class_frequencies(self):
        X, y = toy()
        m = GaussianBayes(reg=0.0).fit(X, y)
        self.assertTrue(np.allclose(np.exp(m.log_priors_), [400 / 900, 300 / 900, 200 / 900]))
        self.assertAlmostEqual(np.exp(m.log_priors_).sum(), 1.0)

    def test_shared_covariance_identical_and_diag_is_diagonal(self):
        X, y = toy()
        s = GaussianBayes(reg=0.01, cov_type="shared").fit(X, y)
        self.assertTrue(np.allclose(s.covs_[0], s.covs_[1]) and np.allclose(s.covs_[1], s.covs_[2]))
        dg = GaussianBayes(reg=0.01, cov_type="diag").fit(X, y)
        for S in dg.covs_:
            self.assertTrue(np.allclose(S, np.diag(np.diag(S))))

    def test_singular_covariance_needs_ridge(self):
        X, y = toy(); X = np.hstack([X, np.zeros((len(X), 1))])   # constant pixel -> singular
        with self.assertRaises(np.linalg.LinAlgError):
            GaussianBayes(reg=0.0).fit(X, y)
        GaussianBayes(reg=1e-2).fit(X, y)                           # succeeds


class TestInference(unittest.TestCase):
    def setUp(self):
        self.X, self.y = toy(); self.m = GaussianBayes(reg=0.0).fit(self.X, self.y)

    def test_log_joint_matches_scipy(self):
        ref = np.column_stack([multivariate_normal(self.m.means_[i], self.m.covs_[i]).logpdf(self.X[:60]) + self.m.log_priors_[i] for i in range(3)])
        self.assertLess(np.abs(ref - self.m.log_joint(self.X[:60])).max(), 1e-9)

    def test_terms_sum_to_log_joint(self):
        T = self.m.terms(self.X[:20])
        self.assertTrue(np.allclose(sum(T.values()), self.m.log_joint(self.X[:20])))

    def test_posteriors_normalised_and_consistent_with_predict(self):
        P = self.m.predict_proba(self.X)
        self.assertTrue(np.allclose(P.sum(1), 1.0)); self.assertTrue((P >= 0).all())
        self.assertTrue((self.m.classes_[P.argmax(1)] == self.m.predict(self.X)).all())

    def test_invariant_to_invertible_affine_map(self):
        rng = np.random.default_rng(3); A = rng.normal(size=(4, 4)) + 3 * np.eye(4); b = rng.normal(size=4)
        m2 = GaussianBayes(reg=0.0).fit(self.X @ A.T + b, self.y)
        self.assertTrue((m2.predict(self.X @ A.T + b) == self.m.predict(self.X)).all())

    def test_label_names_do_not_matter(self):
        m2 = GaussianBayes(reg=0.0).fit(self.X, self.y + 5)
        self.assertTrue((m2.predict(self.X) == self.m.predict(self.X) + 5).all())

    def test_far_away_point_is_numerically_stable(self):
        P = self.m.predict_log_proba(np.full((1, 4), 500.0))
        self.assertTrue(np.isfinite(P).all()); self.assertAlmostEqual(np.exp(P).sum(), 1.0)


class TestDigits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = load_digits(); X, y = d.data / 16.0, d.target
        cls.Xtr, cls.Xte, cls.ytr, cls.yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

    def test_covariance_structure_ordering_and_accuracy(self):
        acc = {ct: (GaussianBayes(r, ct).fit(self.Xtr, self.ytr).predict(self.Xte) == self.yte).mean()
               for ct, r in [("full", 0.01), ("shared", 0.01), ("diag", 0.0316)]}
        self.assertGreaterEqual(acc["full"], 0.985)
        self.assertGreater(acc["full"], acc["shared"]); self.assertGreater(acc["shared"], acc["diag"])

    def test_confusion_matrix_counts(self):
        p = GaussianBayes(0.01).fit(self.Xtr, self.ytr).predict(self.Xte)
        C = confusion(self.yte, p)
        self.assertEqual(C.sum(), len(self.yte)); self.assertEqual(np.trace(C), (p == self.yte).sum())


if __name__ == "__main__":
    unittest.main()
