#include <cmath>
#include <unistd.h>

template<typename T>
SOLE<T>::SOLE(std::size_t n, const Matrix<T>& A, const std::vector<T>& b) : n(n), A(A), b(b) {}


template<typename T>
double norm(const std::vector<T>& vec) {
	double sum = 0;
	for (const auto& elem : vec) {
		sum += elem * elem;
	}
	return std::sqrt(sum);
}


template<typename T>
std::vector<T> operator+(const std::vector<T>& left, const std::vector<T>& right) {
	if (left.size() != right.size()) {
		throw std::runtime_error("Invalid size of operands");
	}

	std::vector<T> res(left);
	for (std::size_t i = 0; i < res.size(); i++) {
		res[i] += right[i];
	}
	return res;
}


template<typename T>
std::vector<T> operator-(const std::vector<T>& left, const std::vector<T>& right) {
	if (left.size() != right.size()) {
		throw std::runtime_error("Invalid size of operands");
	}

	std::vector<T> res(left);
	for (std::size_t i = 0; i < res.size(); i++) {
		res[i] -= right[i];
	}
	return res;
}


template<typename T>
std::vector<T> operator*(const std::vector<T>& vec, const T& scalar) {
	std::vector<T> res(vec);
	for (auto& elem : res) {
		elem *= scalar;
	}
	return res;
}


template<typename T>
std::vector<T> SOLE<T>::SimpleIteration(const std::vector<T>& x0, double eps, const T& tau) const {
	std::vector<T> x(x0);
	std::vector<T> xk(x0);
	while(true) {
		xk = (b - A * x) * tau + x;
		sleep(1);
		std::cout << norm(xk - x) << "\n";
		if (norm(xk - x) < eps) {
			break;
		}
		x = xk;
	}

	return x;
}


template <typename T>
std::vector<T> SOLE<T>::ThomasAlgorithm() const
{
	if (!A.determinant()) {
		throw std::logic_error("matrix is singular\n");
	}
	if (!A[0][0]) {
		throw std::logic_error("vse ploho\n");
	}
	std::vector<T> x(n);
	std::vector<T> alpha(n + 1);
	std::vector<T> beta(n + 1);
	alpha[1] = -A[0][1] / A[0][0];
	beta[1] = b[0] / A[0][0];
	for (std::size_t i = 1; i < n; i++) {
		beta[i+1] = (b[i] - A[i][i - 1] * beta[i]) / (A[i][i - 1] * alpha[i] + A[i][i]);
		alpha[i+1] = -A[i][i + 1] / (A[i][i - 1] * alpha[i] + A[i][i]);
	}
	x[n - 1] = beta[n];
	for (long i = n - 2; i >= 0; i--) {
		x[i] = alpha[i+1]*x[i+1] + beta[i + 1];
	}
	return x;
}


template<typename T>
std::vector<T> SOLE<T>::CholeskySolve() const{
    Matrix<T> C1 = A.getCholeskyDecomposition();
    Matrix<T> C2 = C1.getTransposed();
    std::vector<T> y(n);
    for(std::size_t i = 0; i < n; i++){
        T sum {};
        for(std::size_t j = 0; j < n; j++){
            sum += C1[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / C1[i][i];
    }
    std::vector<T> x(n);
    for(int i = n - 1; i >= 0; i--){
        T sum {};
        for(std::size_t j = i + 1; j < n; j++){
            sum += C2[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / C2[i][i];
    }
    return x;
}


template <typename T>
std::vector<T> SOLE<T>::GaussSolve() const
{
    Matrix<T> augmentedMatrix(A.rowsCount(), A.colsCount() + 1);
    augmentedMatrix = A;
    for (std::size_t i = 0; i < b.size(); i++)
    {
        augmentedMatrix[i].push_back(b[i]);
    }    
    for (std::size_t i = 0; i < augmentedMatrix.rowsCount(); i++)
    {
        std::size_t maxRow = i;
        
        for (std::size_t j = i + 1; j < augmentedMatrix.rowsCount(); j++)
        {
            if (std::abs(augmentedMatrix[j][i]) > std::abs(augmentedMatrix[maxRow][i]))
            {
                maxRow = j;
            }
        }
        if (maxRow != i)
        {
            augmentedMatrix.swapRows(i, maxRow);
        }
        for (std::size_t j = i + 1; j < augmentedMatrix.rowsCount(); j++)
        {
            T factor = augmentedMatrix[j][i] / augmentedMatrix[i][i];
            
            for (std::size_t k = i; k < augmentedMatrix.colsCount(); k++)
            {
                augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
            }
        }
    }
    std::vector<T> x(augmentedMatrix.colsCount() - 1);
    for (int i = augmentedMatrix.rowsCount() - 1; i >= 0; i--)
    {
        T sum = 0;
        for (std::size_t j = i + 1; j < augmentedMatrix.colsCount() - 1; j++)
        {
            sum += augmentedMatrix[i][j] * x[j];
        }
        x[i] = (augmentedMatrix[i][augmentedMatrix.colsCount() - 1] - sum) / augmentedMatrix[i][i];
    }
    return x;
}


template<typename T>
std::vector<T> SOLE<T>::JacobiIteration(const std::vector<T>& x0, double eps) const {
    std::vector<T> x(x0);
    std::vector<T> xk(x0);

    while (true) {
        for (std::size_t i = 0; i < n; i++) {
            T sum = b[i];

            for (std::size_t j = 0; j < n; j++) {
                if (i != j) {
                    sum -= A[i][j] * x[j];
                }
            }

            xk[i] = sum / A[i][i];
        }

        if (norm(xk - x) < eps) {
            break;
        }

        x = xk;
    }

    return x;
}


template<typename T>
std::vector<T> SOLE<T>::LUSolve() const {
    Matrix<T> L(n, n);
    Matrix<T> U(n, n);
    std::vector<T> y(n);
    std::vector<T> x(n);
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            if (i > j) {
                T sum = 0;
                for (std::size_t k = 0; k < j; k++) {
                    sum += L[i][k] * U[k][j];
                }
                L[i][j] = (A[i][j] - sum) / U[j][j];
                U[j][i] = 0;
            } else if (i == j) {
                T sum1 = 0;
                for (std::size_t k = 0; k < i; k++) {
                    sum1 += L[i][k] * U[k][i];
                }
                L[i][i] = 1;
                U[i][i] = A[i][i] - sum1;
            } else {
                T sum2 = 0;
                for (std::size_t k = 0; k < i; k++) {
                    sum2 += L[i][k] * U[k][j];
                }
                U[i][j] = A[i][j] - sum2;
                L[i][j] = 0;
            }
        }
    }
    for (std::size_t i = 0; i < n; i++) {
        T sum = 0;
        for (std::size_t j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }
    for (long i = n - 1; i >= 0; i--) {
        T sum = 0;
        for (std::size_t j = i + 1; j < n; j++) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x;
}


template<typename T>
std::vector<T> SOLE<T>::CramerSolve() const {
    T detA = A.getDet();
    if (detA == 0) {
        throw std::runtime_error("The system is either inconsistent or has infinite solutions.");
    }
    std::vector<T> x(n);
    for (std::size_t i = 0; i < n; i++) {
        Matrix<T> Ai = A;
        for (std::size_t j = 0; j < n; j++) {
            Ai[j][i] = b[j];
        }
        x[i] = Ai.getDet() / detA;
    }

    return x;
}


template<typename T>
std::vector<T> SOLE<T>::InversedSolve() const {
    Matrix<T> invA = A.getInversedMatrix();
    std::vector<T> x = invA * b;
    return x;
}


template<typename T>
std::vector<T> SOLE<T>::SeidelIteration(const std::vector<T>& x0, double eps) const {
    std::vector<T> x(x0);
    std::vector<T> xk(x0);

    while (1) {
        for (std::size_t i = 0; i < n; i++) {
            T sum1 = 0;
            for (std::size_t j = 0; j < i; j++) {
                sum1 += A[i][j] * xk[j];
            }
            T sum2 = 0;
            for (std::size_t j = i + 1; j < n; j++) {
                sum2 += A[i][j] * x[j];
            }

            xk[i] = (b[i] - sum1 - sum2) / A[i][i];
        }

        if (norm(xk - x) < eps) {
            break;
        }
        x = xk;
    }

    return x;
}
