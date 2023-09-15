#include <cstdlib>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>


template <typename T>
Matrix<T> Matrix<T>::IdentityMatrix(std::size_t n)
{
	Matrix<T> res(n, n);
	for (std::size_t i = 0; i < n; i++)
	{
		res.data[i][i] = 1;
	}
	return res;
}


template <typename T>
Matrix<T> Matrix<T>::RotationMatrix(std::size_t n, std::size_t i, std::size_t j, const T &phi)
{
	Matrix<T> res = IdentityMatrix(n);
	res.data[i][i] = std::cos(phi);
	res.data[i][j] = -std::sin(phi);
	res.data[j][i] = std::sin(phi);
	res.data[j][j] = std::cos(phi);
	return res;
}


template <typename T>
Matrix<T>::Matrix() : m(0), n(0) {}


template <typename T>
Matrix<T>::Matrix(std::size_t m, std::size_t n, const T &value) : m(m), n(n)
{
	data = std::vector<std::vector<T>>(m);
	for (std::size_t i{}; i < m; i++)
	{
		data[i] = std::vector<T>(n, value);
	}
}


template <typename T>
Matrix<T>::Matrix(const std::initializer_list<std::initializer_list<T>> &other)
{
	m = other.size();
	data = std::vector<std::vector<T>>();
	for (const auto &row : other)
	{
		data.push_back(row);
	}
	n = data[0].size();
}


template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other) : m(other.m), n(other.n), data(other.data) {}


template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
{
	m = other.m;
	n = other.n;
	data = other.data;
	return *this;
}


template <typename T>
bool Matrix<T>::isSquare() const
{
	if (this->m == this->n)
		return true;
	return false;
}


template <typename T>
bool Matrix<T>::isUpperTriangular() const
{
	if (this->m != this->n)
		return false;

	for (std::size_t i = 1; i < m; i++)
	{
		for (std::size_t j = 0; j < n; j++)
		{
			if (j < i && this->data[i][j] != 0)
				return false;
		}
	}
	return true;
}


template <typename T>
bool Matrix<T>::isLowerTriangular() const
{
	if (this->m != this->n)
		return false;

	for (std::size_t i = 0; i < m; i++)
	{
		for (std::size_t j = i + 1; j < n; j++)
		{
			if (this->data[i][j] != 0)
				return false;
		}
	}
	return true;
}


template <typename T>
bool Matrix<T>::isSymmetric() const
{
	if (this->m != this->n)
		return false;
	for (std::size_t i = 0; i < m; i++)
	{
		for (std::size_t j = 0; j < n; j++)
		{
			if (this->data[i][j] != this->data[j][i])
				return false;
		}
	}
	return true;
}


template <typename T>
bool Matrix<T>::isDiagonal() const
{
	if (this->m != this->n)
		return false;

	for (std::size_t i = 1; i < m; i++)
	{
		for (std::size_t j = 0; j < n; j++)
		{
			if (j != i)
			{
				if (this->data[i][j] != 0)
					return false;
			}
		}
	}
	return true;
}


template <typename T>
bool Matrix<T>::isOrthognal() const
{
	if (this->m != this->n)
		return false;

	Matrix<T> transpose = this->getTransposed();
	Matrix<T> res = (*this) * transpose;

	return res == Matrix<T>::IdentityMatrix(this->m);
}


template <typename T>
bool Matrix<T>::isConjugate() const
{
    for (std::size_t i = 0; i < m; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            if (this->data[i][j] != std::conj(this->data[i][j]))
                return false;
        }
    }
    
    return true;
}


template <typename T>
bool Matrix<T>::isHermitian() const
{
    if (this->m != this->n)
        return false;
    
    for (std::size_t i = 0; i < m; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            if (this->data[i][j] != std::conj(this->data[j][i]))
                return false;
        }
    }
    
    return true;
}


template <typename T>
bool Matrix<T>::isNormal() const
{
    Matrix<T> conjugateTranspose = getConjugateTranspose();
    Matrix<T> selfTranspose = getTransposed(); 
    return (*this * conjugateTranspose) == (conjugateTranspose * selfTranspose);
}


template <typename T>
std::size_t Matrix<T>::rowsCount() const
{
	return m;
}


template <typename T>
std::size_t Matrix<T>::colsCount() const
{
	return n;
}


template <typename T>
std::vector<T> Matrix<T>::getCol(std::size_t index) const
{
	std::vector<T> res;
	for (std::size_t i = 0; i < n; i++)
	{
		res.push_back(data[i][index]);
	}
	return res;
}


template <typename T>
std::vector<T> Matrix<T>::getRow(std::size_t index) const
{
	return data[index];
}


template <typename T>
T Matrix<T>::getTrace() const
{
    T trace = 0;
    for (std::size_t i = 0; i < m; i++)
    {
        trace += data[i][i];
    }
    return trace;
}


template <typename T>
T Matrix<T>::getNorm() const
{
    T norm = 0;
    for (std::size_t i = 0; i < m; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            norm += std::pow(std::abs(data[i][j]), 2);
        }
    }
    return std::sqrt(norm);
}


template <typename T>
T Matrix<T>::getDet() const
{
    if (!isSquare())
    {
        throw std::runtime_error("Determinant is defined only for square matrices.");
    }
    if (m == 1)
    {
        return data[0][0];
    }
    else if (m == 2)
    {
        return (data[0][0] * data[1][1]) - (data[0][1] * data[1][0]);
    }
    else
    {
        T determinant = 0;
        Matrix<T> submatrix(m - 1, m - 1);
        
        for (std::size_t col = 0; col < m; col++)
        {
            std::size_t submatrix_row = 0;
            for (std::size_t row = 1; row < m; row++)
            {
                std::size_t submatrix_col = 0;
                for (std::size_t col2 = 0; col2 < m; col2++)
                {
                    if (col2 != col)
                    {
                        submatrix[submatrix_row][submatrix_col] = data[row][col2];
                        submatrix_col++;
                    }
                }
                submatrix_row++;
            }  
            determinant += (col % 2 == 0 ? 1 : -1) * data[0][col] * submatrix.getDet();
        }
        
        return determinant;
    }
}


template <typename T>
std::vector<T> &Matrix<T>::operator[](std::size_t pos)
{
	return data[pos];
}


template <typename T>
const std::vector<T> &Matrix<T>::operator[](std::size_t pos) const
{
	return data[pos];
}


template <typename T>
void Matrix<T>::appendCol(const std::vector<T> &target)
{
	Matrix<T> temp(this->m, this->n + 1, 0);
	for (std::size_t i = 0; i < this->m; i++)
	{
		for (std::size_t j = 0; j < this->n; j++)
		{
			temp.data[i][j] = this->data[i][j];
		}
	}
	for (std::size_t i = 0; i < temp.m; i++)
	{
		temp.data[i][temp.n - 1] = target[i];
	}
	*this = temp;
}


template <typename T>
void Matrix<T>::appendRow(const std::vector<T> &target)
{
	Matrix<T> temp(this->m + 1, this->n, 0);
	for (std::size_t i = 0; i < this->m; i++)
	{
		for (std::size_t j = 0; j < this->n; j++)
		{
			temp.data[i][j] = this->data[i][j];
		}
	}
	for (std::size_t i = 0; i < temp.m; i++)
	{
		temp.data[temp.m - 1][i] = target[i];
	}
	*this = temp;
}


template <typename T>
void Matrix<T>::swapRows(std::size_t i, std::size_t j)
{
	std::vector<T> temp(data[i]);
	data[i] = data[j];
	data[j] = temp;
}


template <typename T>
void Matrix<T>::swapCols(std::size_t i, std::size_t j)
{
	std::vector<T> temp(m);
	for (std::size_t k = 0; k < m; k++)
	{
		temp[k] = data[k][i];
		data[k][i] = data[k][j];
		data[k][j] = temp[k];
	}
}


template <typename T>
Matrix<T> Matrix<T>::getTransposed() const
{
	Matrix res(n, m);
	for (std::size_t i = 0; i < m; i++)
	{
		for (std::size_t j = 0; j < n; j++)
		{
			res.data[j][i] = data[i][j];
		}
	}
	return res;
}


template <typename T>
Matrix<T> Matrix<T>::getConjugateTranspose() const
{
    Matrix<T> res(n, m);
    
    for (std::size_t i = 0; i < m; i++)
    {
        for (std::size_t j = 0; j < n; j++)
        {
            res[j][i] = std::conj(data[i][j]);
        }
    }
    
    return res;
}


template <typename T>
Matrix<T> Matrix<T>::getUpTrapezoidal() const
{
	Matrix<T> res(*this);
	for (std::size_t i = 1; i < m; i++)
	{
		for (std::size_t j = 0; j < std::min(i, n); j++)
		{
			res.data[i][j] = 0;
		}
	}
	return res;
}


template <typename T>
Matrix<T> Matrix<T>::getDownTrapezoidal() const
{
	Matrix<T> res(*this);
	for (std::size_t i = 0; i < m; i++)
	{
		for (std::size_t j = i + 1; j < n; j++)
		{
			res.data[i][j] = 0;
		}
	}
	return res;
}


template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::getQRDecomposition() const
{
	Matrix R(*this);
	Matrix Q = IdentityMatrix(n);
	std::cout << *this << "\n";
	for (std::size_t j = 0; j < n - 1; j++)
	{
		for (std::size_t i = j + 1; i < n; i++)
		{
			Matrix V = RotationMatrix(n, i, j, std::atan2(R.data[i][j], R.data[j][j]));
			R = V * R;
			Q = V * Q;
		}
	}
	Q = Q.getTransposed();
	std::cout << Q << "\n"
			  << R << "\n";
	std::cout << Q * R << "\n";
	return std::make_pair(R, R);
}


template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::getLUDecomposition() const
{
	Matrix<T> L = IdentityMatrix(m);
	Matrix<T> U(*this);
	for (std::size_t k = 0; k < n - 1; k++)
	{
		for (std::size_t i = k + 1; i < n; i++)
		{
			if (U[k][k] == 0)
			{
				throw std::runtime_error("Matrix is not LU decomposable");
			}
			T factor = U[i][k] / U[k][k];
			L[i][k] = factor;
			for (std::size_t j = k; j < n; j++)
			{
				U[i][j] -= factor * U[k][j];
			}
		}
	}

	return std::make_pair(L, U);
}


template <typename T>
std::pair<Matrix<T>, Matrix<T>> Matrix<T>::getCholeskyDecomposition() const
{
    Matrix<T> L(m, m);
    for (std::size_t i = 0; i < m; i++)
    {
        for (std::size_t j = 0; j <= i; j++)
        {
            T sum = 0;
            if (j == i)
            {
                for (std::size_t k = 0; k < j; k++)
                {
                    sum += L[j][k] * L[j][k];
                }
                L[j][j] = std::sqrt(data[j][j] - sum);
            }
            else
            {
                for (std::size_t k = 0; k < j; k++)
                {
                    sum += L[i][k] * L[j][k];
                }
                L[i][j] = (data[i][j] - sum) / L[j][j];
            }
        }
    }

    return std::make_pair(L, L.getTransposed());
}


template <typename T>
Matrix<T> Matrix<T>::getInversedMatrix() const
{
    Matrix<T> identity = IdentityMatrix(m);
    Matrix<T> augmented = *this;
    for (std::size_t i = 0; i < m; i++)
    {
        std::size_t maxRow = i;
        for (std::size_t j = i + 1; j < m; j++)
        {
            if (std::abs(augmented[j][i]) > std::abs(augmented[maxRow][i]))
            {
                maxRow = j;
            }
        }
        if (maxRow != i)
        {
            augmented.swapRows(i, maxRow);
            identity.swapRows(i, maxRow);
        }
        for (std::size_t j = 0; j < m; j++)
        {
            if (j != i)
            {
                T factor = augmented[j][i];
                for (std::size_t k = i; k < m; k++)
                {
                    augmented[j][k] -= factor * augmented[i][k];
                }
                for (std::size_t k = 0; k < m; k++)
                {
                    identity[j][k] -= factor * identity[i][k];
                }
            }
        }
    }

    return identity;
}


template <typename T>
Matrix<T> operator*(const Matrix<T> &left, const Matrix<T> &right)
{
	if (left.n != right.m)
	{
		throw std::runtime_error("Invlid size of operands");
	}

	Matrix<T> res(left.n, right.m);
	for (std::size_t i = 0; i < left.m; i++)
	{
		for (std::size_t j = 0; j < right.n; j++)
		{
			for (std::size_t k = 0; k < left.n; k++)
			{
				res.data[i][j] += left.data[i][k] * right.data[k][j];
			}
		}
	}
	return res;
}


template <typename T>
std::vector<T> operator*(const Matrix<T> &mat, const std::vector<T> &vec)
{
	if (mat.n != vec.size())
	{
		throw std::runtime_error("Invalid size of operands");
	}

	std::vector<T> res(vec.size());

	for (std::size_t i = 0; i < mat.m; i++)
	{
		for (std::size_t j = 0; j < mat.n; j++)
		{
			res[i] += mat.data[i][j] * vec[j];
		}
	}
	return res;
}


template <typename V>
std::vector<V> operator*(const std::vector<V> &vec, const Matrix<V> &mat)
{
	if (vec.size() != mat.m)
	{
		throw std::runtime_error("Invalid size of operands");
	}

	std::vector<V> res(mat.n);

	for (std::size_t j = 0; j < mat.n; j++)
	{
		for (std::size_t i = 0; i < mat.m; i++)
		{
			res[j] += vec[i] * mat.data[i][j];
		}
	}

	return res;
}


template <typename T>
Matrix<T> operator+(const Matrix<T> &matrix_left, const Matrix<T> &matrix_right)
{
	if (matrix_right.m != matrix_left.m || matrix_right.n != matrix_left.n)
	{
		throw std::runtime_error("Invalid size of operands");
	}

	Matrix<T> res(matrix_right.m, matrix_right.n);

	for (std::size_t j = 0; j < matrix_right.m; j++)
	{
		for (std::size_t i = 0; i < matrix_right.n; i++)
		{
			res.data[i][j] = matrix_left.data[i][j] + matrix_right.data[i][j];
		}
	}

	return res;
}


template <typename T>
Matrix<T> operator-(const Matrix<T> &matrix_left, const Matrix<T> &matrix_right)
{
	if (matrix_right.m != matrix_left.m || matrix_right.n != matrix_left.n)
	{
		throw std::runtime_error("Invalid size of operands");
	}

	Matrix<T> res(matrix_right.m, matrix_right.n);

	for (std::size_t j = 0; j < matrix_right.m; j++)
	{
		for (std::size_t i = 0; i < matrix_right.n; i++)
		{
			res.data[i][j] = matrix_left.data[i][j] - matrix_right.data[i][j];
		}
	}

	return res;
}


template <typename V>
Matrix<V> operator*(const Matrix<V> &self, const V &var)
{
	Matrix<V> res(self.m, self.n);

	for (std::size_t j = 0; j < self.m; j++)
	{
		for (std::size_t i = 0; i < self.n; i++)
		{
			res.data[i][j] = self.data[i][j] * var;
		}
	}
	return res;
}


template <typename V>
Matrix<V> operator*(const V &var, const Matrix<V> &self)
{
	Matrix<V> res(self.m, self.n);

	for (std::size_t j = 0; j < self.m; j++)
	{
		for (std::size_t i = 0; i < self.n; i++)
		{
			res.data[i][j] = self.data[i][j] * var;
		}
	}
	return res;
}


template <typename V>
Matrix<V> operator/(const Matrix<V> &self, const V &var)
{
	Matrix<V> res(self.m, self.n);

	for (std::size_t j = 0; j < self.m; j++)
	{
		for (std::size_t i = 0; i < self.n; i++)
		{
			res.data[i][j] = self.data[i][j] / var;
		}
	}
	return res;
}


template <typename T>
std::istream &operator>>(std::istream &in, Matrix<T> &self)
{
	for (auto &row : self.data)
	{
		for (auto &elem : row)
		{
			in >> elem;
		}
	}
	return in;
}


template <typename T>
std::ostream &operator<<(std::ostream &out, const Matrix<T> &self)
{
	for (const auto &row : self.data)
	{
		for (const auto &elem : row)
		{
			out << elem << " ";
		}
		out << std::endl;
	}
	return out;
}


template <typename T>
void printVector(const std::vector<T> &vec)
{
	for (const auto &element : vec)
	{
		std::cout << element << " ";
	}
	std::cout << std::endl;
}


template <typename V>
bool operator==(const Matrix<V> &matrix_left, const Matrix<V> &matrix_right)
{
	if (matrix_left.m != matrix_right.m || matrix_left.n != matrix_right.n)
		return false;

	for (std::size_t i = 0; i < matrix_left.m; i++)
	{
		for (std::size_t j = 0; j < matrix_left.n; j++)
		{
			if (matrix_left.data[i][j] != matrix_right.data[i][j])
				return false;
		}
	}

	return true;
}


template <typename V>
bool operator!=(const Matrix<V> &matrix_left, const Matrix<V> &matrix_right)
{
	if (matrix_left.m != matrix_right.m || matrix_left.n != matrix_right.n)
		return true;

	for (std::size_t i = 0; i < matrix_left.m; i++)
	{
		for (std::size_t j = 0; j < matrix_left.n; j++)
		{
			if (matrix_left.data[i][j] != matrix_right.data[i][j])
				return true;
		}
	}

	return false;
}