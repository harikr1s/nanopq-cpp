# from the nanopg repo with small changes

import numpy as np
from scipy.cluster.vq import kmeans2, vq
import os


def dist_l2(q, x):
    return np.linalg.norm(q - x, ord=2, axis=1) ** 2


def dist_ip(q, x):
    return q @ x.T


metric_function_map = {"l2": dist_l2, "dot": dist_ip}


class PQ(object):
    def __init__(self, M, Ks=256, metric="l2", verbose=True):
        assert 0 < Ks <= 2**32
        assert metric in ["l2", "dot"]
        self.M, self.Ks, self.metric, self.verbose = M, Ks, metric, verbose
        self.code_dtype = (
            np.uint8 if Ks <= 2**8 else (np.uint16 if Ks <=
                                         2**16 else np.uint32)
        )
        self.codewords = None
        self.Ds = None

        if verbose:
            print(
                "M: {}, Ks: {}, metric : {}, code_dtype: {}".format(
                    M, Ks, self.code_dtype, metric
                )
            )

    def __eq__(self, other):
        if isinstance(other, PQ):
            return (
                self.M,
                self.Ks,
                self.metric,
                self.verbose,
                self.code_dtype,
                self.Ds,
            ) == (
                other.M,
                other.Ks,
                other.metric,
                other.verbose,
                other.code_dtype,
                other.Ds,
            ) and np.array_equal(
                self.codewords, other.codewords
            )
        else:
            return False

    def fit(self, vecs, iter=20, seed=123, minit="points"):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert self.Ks < N, "the number of training vector should be more than Ks"
        assert D % self.M == 0, "input dimension must be dividable by M"
        assert minit in ["random", "++", "points", "matrix"]
        self.Ds = int(D / self.M)

        np.random.seed(seed)
        if self.verbose:
            print("iter: {}, seed: {}".format(iter, seed))

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self.codewords = np.zeros((self.M, self.Ks, self.Ds), dtype=np.float32)
        # self.allvecs_sub = []
        for m in range(self.M):
            if self.verbose:
                print("Training the subspace: {} / {}".format(m+1, self.M))
            self.vecs_sub = vecs[:, m * self.Ds: (m + 1) * self.Ds]
            # self.allvecs_sub.append(self.vecs_sub)
            self.codewords[m], _ = kmeans2(
                self.vecs_sub, self.Ks, iter=iter, minit=minit)
        return self

    def encode(self, vecs):
        assert vecs.dtype == np.float32
        assert vecs.ndim == 2
        N, D = vecs.shape
        assert D == self.Ds * self.M, "input dimension must be Ds * M"

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.M), dtype=self.code_dtype)
        for m in range(self.M):
            if self.verbose:
                print("Encoding the subspace: {} / {}".format(m+1, self.M))
            vecs_sub = vecs[:, m * self.Ds: (m + 1) * self.Ds]
            codes[:, m], _ = vq(vecs_sub, self.codewords[m])

        return codes


pq = PQ(M=4)
vecs = np.random.random((300, 12)).astype(np.float32)
pq.fit(vecs)

if pq.verbose:
    v = 1
else:
    v = 0

os.makedirs("./out/", exist_ok=True)

with open("./out/variables.txt", "w") as variable_file:
    variable_file.write(
        f"{pq.Ds}\n{pq.M}\n{v}\n{pq.codewords.shape[0]}\n{pq.codewords.shape[1]}\n{pq.codewords.shape[2]}")

assert vecs.dtype == np.float32
assert vecs.ndim == 2
N, D = vecs.shape
assert D == pq.Ds * pq.M, "input dimension must be Ds * M"

# saving 'vecs_sub'
np.savetxt("./out/vecs_sub.txt", pq.vecs_sub, delimiter=" ")

# saving input vectors 'vecs'
np.savetxt("./out/vecs.txt", vecs, delimiter=" ")

# saving 'codewords' as .txt after reshaping it to 2D first, as 3D arrays can't be converted to .txt
# np.savetxt("./out/codewords.txt",
#            pq.codewords.reshape(pq.codewords.shape[0], -1))
np.save("out/codewords", pq.codewords)

# saving an uninitialized np array 'codes'
codes = np.empty((N, pq.M), dtype=pq.code_dtype)
np.savetxt("./out/codes.txt", codes, delimiter=" ")

encode_out = pq.encode(vecs)
np.savetxt("./out/py_encoded.txt", encode_out, fmt='%d', delimiter=" ")
