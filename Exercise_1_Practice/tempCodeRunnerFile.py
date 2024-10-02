A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.
        A[-1] = A[0] + A[-2]
        b = np.ones(10)