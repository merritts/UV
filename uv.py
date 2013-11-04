import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

def rmse(util_mtx, p, q):
    e = 0.
    m = 0.
    r,c = util_mtx.shape
    for i in xrange(r):
        for j in util_mtx[i].indices:
            e += (util_mtx[i,j]-np.dot(q[j], p[i]))**2
            m+=1
    return np.sqrt(e/m)

def sgd_uv(util_mtx, f=5, lr=0.001, reg=0.1):
    err_arr = []
    r,c = util_mtx.shape
    #item matrix
    q = np.random.rand(c,f)
    #user matrix
    p = np.random.rand(r,f)
    #fit the matrix
    for c in xrange(1000):
        for i in xrange(r):
            for j in util_mtx[i].indices:
                err = util_mtx[i,j] - np.dot(q[j], p[i])
                q[j] = q[j] + lr*(err*p[i]-reg*q[j])
                p[i] = p[i] + lr*(err*q[j]-reg*p[i])
        err_arr.append(rmse(util_mtx,p,q))
    return p,q,err_arr

if __name__=="__main__":
    A = scipy.sparse.lil_matrix((100, 100))
    A[0, :10] = np.random.rand(10)
    A[1, 10:20] = A[0, :10]
    A.setdiag(np.random.rand(100))
    A = A.tocsr()
    err = []
    for j in range(5,10):
        p,q,err_arr = sgd_uv(A, f=j, lr=0.001, reg=0.1)
        plt.plot(err_arr, linewidth=2)
        plt.xlabel('Iteration, i', fontsize=20)
        plt.ylabel('RMSE', fontsize=20)
        plt.title('Components = '+str(j), fontsize=20)
        plt.show()
        err.append(err_arr[-1])
    plt.plot(err, linewidth=2)
    plt.xlabel('Component size, f', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    plt.show()