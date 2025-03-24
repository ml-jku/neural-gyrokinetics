from typing import Sequence
import numpy as np


class RunningMeanStd:
    def __init__(self, shape: Sequence[int], epsilon: float = 1e-4):
        """
        Calculates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float32)
        self.min = np.zeros(shape, np.float32)
        self.max = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(
            other.mean, other.var, other.min, other.max, other.count
        )

    def update(self, mean, var, min, max, count=1.) -> None:
        self.update_from_moments(mean, var, min, max, count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_min: np.ndarray,
        batch_max: np.ndarray,
        batch_count: float = 1.0,
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)
        self.mean = new_mean
        self.var = new_var
        self.count = new_count
