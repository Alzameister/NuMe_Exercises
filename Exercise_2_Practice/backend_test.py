import numpy as np
import sys
import traceback
import backend


class Tester:
    def __init__(self):
        self.module = None
        self.runtime = 300

    def testL(self, L):
        # Lower Triangle
        if (not (np.abs(L - np.tril(L)) < 1e-6).all()):
            return False
        return True

    def testU(self, U):
        # Upper Triangle
        if (not (np.abs(U - np.triu(U)) < 1e-6).all()):
            return False
        return True

    def testP(self, P):
        # Square
        if (not P.shape[0] == P.shape[1]):
            return False
        # Only 0 and 1
        if (not (np.unique(P) == [0, 1]).all()):
            return False
        # Amount of 1s has to be equal to rows/columns
        if (not np.count_nonzero(P) == P.shape[0]):
            return False
        # Only one 1 per row/column
        for i in range(P.shape[0]):
            if (np.count_nonzero(P[i]) != 1 or
                    np.count_nonzero(P[:, i]) != 1):
                return False
        # Determinant has to be 1
        if (np.abs(np.linalg.det(P)) - 1 > 1e-16):
            return False
        return True

    def testPLU(self, P, L, U, A):
        passed = True
        additionalComments = ""
        if (not self.testP(P)):
            additionalComments += "P failed. Partial pivoting is not implemented"
            passed = False
        if (not self.testL(L)):
            additionalComments += "L failed. "
            passed = False
        if (not self.testU(U)):
            additionalComments += "U failed. "
            passed = False
        if (not (np.abs(P.dot(A) - L.dot(U)) < 1e-6).all()):
            additionalComments += "result imprecise. "
            passed = False
        elif passed:
            additionalComments += "passed. "
        return passed, additionalComments

    #############################################
    # Task a
    #############################################

    def testA(self, l: list, task):
        comments = ""

        def evaluate(A):
            nonlocal comments
            try:
                P, L, U = self.module.lu(np.copy(A))
                passed, additionalComments = self.testPLU(P, L, U, A)
                comments += additionalComments + " "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # 10x10 upper triangular
        comments += "10x10 upper triangle case "

        A = np.triu(np.ones((10, 10)))
        evaluate(A)

        # 10x10 floats
        comments += "10x10 case "

        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.

        evaluate(A)

        # Pivoting case
        comments += "10x10 Pivoting case "

        A = np.triu(np.ones((10, 10)))
        A = np.roll(A, 1, axis=0)

        evaluate(A)
        result = [task, comments]
        print(result)
        l.extend(result)

    #############################################
    # Task b
    #############################################

    def testB(self, l: list, task):
        comments = ""

        def evaluate(A):
            nonlocal comments
            try:
                reference = np.linalg.det(A)
                det = self.module.determinant(np.copy(A))
                if (np.abs(reference - det) < 1e-6):
                    comments += "passed."
                else:
                    comments += "failed."
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # 10x10 Identity
        comments += "Identity case "

        A = np.identity(10)

        evaluate(A)

        # 10x10 Zeros
        comments += "Zero case "

        A = np.zeros((10, 10))

        evaluate(A)

        # 10x10 Ones
        comments += "Ones case "

        A = np.ones((10, 10))

        evaluate(A)

        # 10x10 floats
        comments += "10x10 case "

        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.
            A[:, i] *= (10 - i + 1) / 10.
        A = A.transpose().dot(A).dot(A) * np.pi / np.e * 50.

        evaluate(A)
        result = [task, comments]
        print(result)
        l.extend(result)

    def performTest(self, func, task):
        l = []
        try:
            func(l, task)
            return l
        except Exception as e:
            return []

    def runTests(self, module, l):
        self.module = module

        def evaluateResult(task, result):
            if (len(result) == 0):
                l.append([task, 0, "Interrupt."])
            else:
                l.append(result)

        result = self.performTest(self.testA, "2.1a)")
        evaluateResult("2.1a)", result)

        result = self.performTest(self.testB, "2.1b)")
        evaluateResult("2.1b)", result)

        return l


tester = Tester()
overall_result = []
tester.runTests(backend, overall_result)