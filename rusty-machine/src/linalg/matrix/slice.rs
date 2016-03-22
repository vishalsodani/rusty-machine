use super::Matrix;

use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero};

use linalg::utils;

/// A MatrixSlice
///
/// This struct provides a slice into a matrix.
///
/// The struct contains the upper left point of the slice
/// and the width and height of the slice.
#[derive(Debug, Clone, Copy)]
pub struct MatrixSlice<'a, T: 'a> {
    window_corner: [usize; 2],
    rows: usize,
    cols: usize,
    mat: &'a Matrix<T>,
}

impl<'a, T> MatrixSlice<'a, T> {
	/// Return the number of rows in the slice.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the number of columns in the slice.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Produce a matrix slice from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &'a Matrix<T>,
                       start: [usize; 2],
                       rows: usize,
                       cols: usize)
                       -> MatrixSlice<'a, T> {
        assert!(start[0] + rows <= mat.rows(),
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= mat.cols(),
                "View dimensions exceed matrix dimensions.");

        MatrixSlice {
            window_corner: start,
            rows: rows,
            cols: cols,
            mat: mat,
        }
    }

    /// Returns an iterator over the matrix slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::slice::MatrixSlice;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let slice = MatrixSlice::from_matrix(&a, [1,1], 2, 2);
    ///
    /// let slice_data = slice.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(slice_data, vec![4,5,7,8]);
    /// ```
    pub fn iter(&'a self) -> SliceIter<'a, T> {
        SliceIter {
            slice: self,
            row_pos: 0,
            col_pos: 0,
        }
    }
}

impl<'a, T: Copy> MatrixSlice<'a, T> {
	/// Convert the matrix slice into a new Matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        let slice_data = self.iter().map(|v| *v).collect::<Vec<T>>();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: slice_data,
        }
    }
}

/// Iterator for the MatrixSlice
///
/// Iterates over the underlying slice data
/// in row-major order.
#[derive(Debug)]
pub struct SliceIter<'a, T: 'a> {
    slice: &'a MatrixSlice<'a, T>,
    row_pos: usize,
    col_pos: usize,
}

impl<'a, T> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let raw_row_idx = self.slice.window_corner[0] + self.row_pos;
        let raw_col_idx = self.slice.window_corner[1] + self.col_pos;

        // Set the position of the next element
        if self.row_pos < self.slice.rows {
            // If end of row, set to start of next row
            if self.col_pos == self.slice.cols - 1 {
                self.row_pos += 1usize;
                self.col_pos = 0usize;
            } else {
                self.col_pos += 1usize;
            }

            Some(&self.slice.mat[[raw_row_idx, raw_col_idx]])
        } else {
            None
        }
    }
}

/// Indexes matrix.
///
/// Takes row index first then column.
impl<'a, T> Index<[usize; 2]> for MatrixSlice<'a, T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &self.mat.data().get_unchecked((self.window_corner[0] + idx[0]) * self.mat.cols +
                                           self.window_corner[1] +
                                           idx[1])
        }
    }
}

/// Multiplies matrix by scalar.
impl<'a, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        (&self) * (&f)
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'b T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        (&self) * f
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        self * (&f)
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, 'c, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'c T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.iter().map(|v| (*v) * (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Multiplies matrix by matrix.
impl<'a, T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<Matrix<T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'b Matrix<T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix by matrix.
impl<'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'c Matrix<T>> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.rows, "Matrix dimensions do not agree.");

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        unsafe {
            for i in 0..self.rows
            {
                for k in 0..m.rows
                {
                    for j in 0..m.cols
                    {
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) + *self.mat.data().get_unchecked((self.window_corner[0] + i) *self.cols + (self.window_corner[1] + k)) * *m.data().get_unchecked(k*m.cols + j);
                    }
                }
            }
        }

        Matrix {
            rows: self.rows,
            cols: m.cols,
            data: new_data
        }
    }
}

/// Multiplies matrix by matrix.
impl<'a, 'b, T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<MatrixSlice<'b, T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: MatrixSlice<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<MatrixSlice<'b, T>> for &'c MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: MatrixSlice<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'c MatrixSlice<'b, T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &MatrixSlice<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix by matrix.
impl<'a, 'b, 'c, 'd, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'d MatrixSlice<'b, T>> for &'c MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &MatrixSlice<T>) -> Matrix<T> {
        assert!(self.cols == m.rows, "Matrix dimensions do not agree.");

        let mut new_data = vec![T::zero(); self.rows * m.cols];

        unsafe {
            for i in 0..self.rows
            {
                for k in 0..m.rows
                {
                    for j in 0..m.cols
                    {
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) +
                        							*self.mat.data().get_unchecked((self.window_corner[0] + i) * self.cols + self.window_corner[1] + k) *
                        							*m.mat.data().get_unchecked((m.window_corner[0] + k)*m.cols + m.window_corner[1] + j);
                    }
                }
            }
        }

        Matrix {
            rows: self.rows,
            cols: m.cols,
            data: new_data
        }
    }
}

/// Adds scalar to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        (&self) + f
    }
}

impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        let new_data = self.iter().map(|v| (*v) + (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix to matrix.
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b Matrix<T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c Matrix<T>> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().map(|x| *x).collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = *x + y });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<MatrixSlice<'b, T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: MatrixSlice<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<MatrixSlice<'b, T>> for &'c MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: MatrixSlice<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c MatrixSlice<'b, T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &MatrixSlice<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, 'd, T: Copy + One + Zero + Add<T, Output = T>> Add<&'d MatrixSlice<'b, T>> for &'c MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn add(self, m: &MatrixSlice<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = self.iter().zip(m.iter()).map(|(u, v)| *u + *v).collect();
        

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts scalar from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'a T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        let new_data = self.iter().map(|v| (*v) - *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts matrix from matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'b Matrix<T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c Matrix<T>> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().map(|x| *x).collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = *x - y });
        

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<MatrixSlice<'b, T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: MatrixSlice<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<MatrixSlice<'b, T>> for &'c MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: MatrixSlice<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c MatrixSlice<'b, T>> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &MatrixSlice<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts matrix from matrix.
impl<'a, 'b, 'c, 'd, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'d MatrixSlice<'b, T>> for &'c MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, m: &MatrixSlice<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = self.iter().zip(m.iter()).map(|(u, v)| *u - *v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Divides matrix by scalar.
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        (&self) / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        self / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'b T> for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        (&self) / f
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, 'c, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'c T> for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        assert!(*f != T::zero());

        let new_data = self.iter().map(|v| (*v) / *f).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Gets negative of matrix.
impl<'a, T: Neg<Output = T> + Copy> Neg for MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Gets negative of matrix.
impl<'a, 'b, T: Neg<Output = T> + Copy> Neg for &'b MatrixSlice<'a, T> {
    type Output = Matrix<T>;

    fn neg(self) -> Matrix<T> {
        let new_data = self.iter().map(|v| -*v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

#[cfg(test)]
mod tests {
	use super::MatrixSlice;
	use super::super::Matrix;

	#[test]
	#[should_panic]
	fn make_slice() {
		let a = Matrix::new(3,3, vec![2.0; 9]);
		let _ = MatrixSlice::from_matrix(&a, [1,1], 3, 2);
	}

	#[test]
	fn add_slice() {
		let a = 3.0;
		let b = Matrix::new(3,3, vec![2.0; 9]);
		let c = Matrix::new(2,2, vec![1.0; 4]);

		let d = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m_1 = &d + a;
		assert_eq!(m_1.into_vec(), vec![5.0; 4]);

		let m_2 = &d + c;
		assert_eq!(m_2.into_vec(), vec![3.0; 4]);

		let m_3 = &d + &d;
		assert_eq!(m_3.into_vec(), vec![4.0; 4]);		
	}

	#[test]
	fn sub_slice() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let c = Matrix::new(3,3, vec![2.0; 9]);

		let d = MatrixSlice::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d - a;
		assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

		let m_2 = &d - b;
		assert_eq!(m_2.into_vec(), vec![1.0; 4]);

		let m_3 = &d - &d;
		assert_eq!(m_3.into_vec(), vec![0.0; 4]);
	}

	#[test]
	fn mul_slice() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let c = Matrix::new(3,3, vec![2.0; 9]);

		let d= MatrixSlice::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d * a;
		assert_eq!(m_1.into_vec(), vec![6.0; 4]);

		let m_2 = &d * b;
		assert_eq!(m_2.into_vec(), vec![4.0; 4]);

		let m_3 = &d * d;
		assert_eq!(m_3.into_vec(), vec![8.0; 4]);
	}

	#[test]
	fn div_slice() {
		let a = 3.0;

		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m = c / a;
		assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
	}

	#[test]
	fn neg_slice() {
		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);

		let m = -c;
		assert_eq!(m.into_vec(), vec![-2.0;4]);
	}

	#[test]
	fn index_slice() {
		let b = Matrix::new(3,3, (0..9).collect());

		let c = MatrixSlice::from_matrix(&b, [1,1], 2, 2);
		
		assert_eq!(c[[0,0]], 4);
		assert_eq!(c[[0,1]], 5);
		assert_eq!(c[[1,0]], 7);
		assert_eq!(c[[1,1]], 8);
	}
}