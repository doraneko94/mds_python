# mds_python
Implementing deterministic MDS (Multi Dimensional Scaling) in Python.

# Requirement

## for all
- numpy

## for demo
- matplotlib
- scikit-learn

# How to use
```python
    from mds import MDS
    import numpy as np

    mds = MDS()
    x = np.array([
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 0],
        [3, 7, 1],
    ])
    z = mds.fit_transform(x)
    print(z)
    # array([
    #   [-1.33993844, -0.50090235],
    #   [-1.43485236,  0.46776821],
    #   [ 1.43485236, -0.46776821],
    #   [ 1.33993844,  0.50090235]
    # ])
```

# Theoritical BackGround
MDS transforms the given matrix `X` into a low-dimensional 
matrix `Z` based on the distance between each element.

The distances between the elements are stored in the dissimilarity 
matrix `D`.

## Dissimilarity

In this class `MDS`, the following methods can be used to calculate 
dissimilarity.

### euclidean
![\begin{align*}
d^{Euclidean}_{i,j}=\sqrt{(\mathbf{x}_{i}-\mathbf{x}_{j})^{2}}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Ad%5E%7BEuclidean%7D_%7Bi%2Cj%7D%3D%5Csqrt%7B%28%5Cmathbf%7Bx%7D_%7Bi%7D-%5Cmathbf%7Bx%7D_%7Bj%7D%29%5E%7B2%7D%7D%0A%5Cend%7Balign%2A%7D)

### cosine
![\begin{align*}
d^{Cosine}_{i,j}=1-\frac{\mathbf{x}_{i}^{\mathrm{T}}\mathbf{x}_{j}}{\sqrt{\mathbf{x}_{i}^{\mathrm{T}}\mathbf{x}_{i}}\sqrt{\mathbf{x}_{j}^{\mathrm{T}}\mathbf{x}_{j}}}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Ad%5E%7BCosine%7D_%7Bi%2Cj%7D%3D1-%5Cfrac%7B%5Cmathbf%7Bx%7D_%7Bi%7D%5E%7B%5Cmathrm%7BT%7D%7D%5Cmathbf%7Bx%7D_%7Bj%7D%7D%7B%5Csqrt%7B%5Cmathbf%7Bx%7D_%7Bi%7D%5E%7B%5Cmathrm%7BT%7D%7D%5Cmathbf%7Bx%7D_%7Bi%7D%7D%5Csqrt%7B%5Cmathbf%7Bx%7D_%7Bj%7D%5E%7B%5Cmathrm%7BT%7D%7D%5Cmathbf%7Bx%7D_%7Bj%7D%7D%7D%0A%5Cend%7Balign%2A%7D)

### precomputed
In this mode, any dissimilarity matrix `D` can be used.

To do so, pass the dissimilarity matrix `D` obtained by any method 
as the argument `X` of function `fit` or `fit_transform`.

## Calculation
Suppose that the kernel matrix `K`, which can be represented as 

![\begin{align*}
\mathbf{K}=\mathbf{Z}^{\mathrm{T}}\mathbf{Z}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cmathbf%7BK%7D%3D%5Cmathbf%7BZ%7D%5E%7B%5Cmathrm%7BT%7D%7D%5Cmathbf%7BZ%7D%0A%5Cend%7Balign%2A%7D)

using transformed matrix `Z`, can be obtained from the dissimilarity 
matrix `D`.

`K` is expressed using `D` as

![\begin{align*}
\mathbf{K}=-\frac{1}{2}\mathbf{H}\mathbf{D}\mathbf{H}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cmathbf%7BK%7D%3D-%5Cfrac%7B1%7D%7B2%7D%5Cmathbf%7BH%7D%5Cmathbf%7BD%7D%5Cmathbf%7BH%7D%0A%5Cend%7Balign%2A%7D)

where `H` is the centralization matrix and is defined as 

![\begin{align*}
\mathbf{H}=\mathbf{I}_{n}-\frac{1}{n}\mathbf{J}_{n}.
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cmathbf%7BH%7D%3D%5Cmathbf%7BI%7D_%7Bn%7D-%5Cfrac%7B1%7D%7Bn%7D%5Cmathbf%7BJ%7D_%7Bn%7D.%0A%5Cend%7Balign%2A%7D%0A)

`I_n` is an identity matrix of size `n`, and `J_n` is a square matrix 
of size `n` and all elements are `1`.

Finally, `K` is eigen-decomposed as 

![\begin{align*}
\mathbf{K}=\mathbf{V}\mathbf{L}\mathbf{V}^{\mathrm{T}}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cmathbf%7BK%7D%3D%5Cmathbf%7BV%7D%5Cmathbf%7BL%7D%5Cmathbf%7BV%7D%5E%7B%5Cmathrm%7BT%7D%7D%0A%5Cend%7Balign%2A%7D%0A)

where `L` is a matrix with K's eigenvalues in its diagonal elements, 
and `V` is a matrix consisting of eigenvectors.

Therefore, `Z` can be calculated as 

![\begin{align*}
\mathbf{Z}=\mathrm{sqrt}(\mathbf{L})\mathbf{V}^{\mathrm{T}}
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cmathbf%7BZ%7D%3D%5Cmathrm%7Bsqrt%7D%28%5Cmathbf%7BL%7D%29%5Cmathbf%7BV%7D%5E%7B%5Cmathrm%7BT%7D%7D%0A%5Cend%7Balign%2A%7D%0A)

where `sqrt(A)` represents the operation of taking the square root for all 
the elements of the matrix `A`.

If `L` and `V` are taken to contain only the top `k` eigenvalues/vectors, 
the dimension of `Z` can be reduced to `k`.

# Examples
In the demo in the source code, MDS is applied to the `iris` dataset.

## euclidean
![Euclidean](img/euclidean.png)

## cosine
![Euclidean](img/cosine.png)