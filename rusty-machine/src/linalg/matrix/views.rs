use super::Matrix;

use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero};

/// A MatrixView
///
/// This struct provides a view into a matrix.
///
/// The struct contains the upper left point of the view
/// and the width and height of the view.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a, T: 'a> {
    window_corner: [usize; 2],
    rows: usize,
    cols: usize,
    mat: &'a Matrix<T>,
}

impl<'a, T> MatrixView<'a, T> {
	/// Return the number of rows in the view.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return the number of columns in the view.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Produce a matrix view from a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::views::MatrixView;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let view = MatrixView::from_matrix(&a, [1,1], 2, 2);
    /// ```
    pub fn from_matrix(mat: &'a Matrix<T>,
                       start: [usize; 2],
                       rows: usize,
                       cols: usize)
                       -> MatrixView<'a, T> {
        assert!(start[0] + rows <= mat.rows(),
                "View dimensions exceed matrix dimensions.");
        assert!(start[1] + cols <= mat.cols(),
                "View dimensions exceed matrix dimensions.");

        MatrixView {
            window_corner: start,
            rows: rows,
            cols: cols,
            mat: mat,
        }
    }

    /// Returns an iterator over the matrix view.
    ///
    /// # Examples
    ///
    /// ```
    /// use rusty_machine::linalg::matrix::Matrix;
    /// use rusty_machine::linalg::matrix::views::MatrixView;
    ///
    /// let a = Matrix::new(3,3, (0..9).collect::<Vec<usize>>());
    /// let view = MatrixView::from_matrix(&a, [1,1], 2, 2);
    ///
    /// let view_data = view.iter().map(|v| *v).collect::<Vec<usize>>();
    /// assert_eq!(view_data, vec![4,5,7,8]);
    /// ```
    pub fn iter(&'a self) -> ViewIter<'a, T> {
        ViewIter {
            view: self,
            row_pos: 0,
            col_pos: 0,
        }
    }
}

impl<'a, T: Copy> MatrixView<'a, T> {
	/// Convert the matrix view into a new Matrix.
    pub fn into_matrix(self) -> Matrix<T> {
        let view_data = self.iter().map(|v| *v).collect::<Vec<T>>();
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: view_data,
        }
    }
}

/// Iterator for the MatrixView
///
/// Iterates over the underlying view data
/// in row-major order.
#[derive(Debug)]
pub struct ViewIter<'a, T: 'a> {
    view: &'a MatrixView<'a, T>,
    row_pos: usize,
    col_pos: usize,
}

impl<'a, T> Iterator for ViewIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let raw_row_idx = self.view.window_corner[0] + self.row_pos;
        let raw_col_idx = self.view.window_corner[1] + self.col_pos;

        // Set the position of the next element
        if self.row_pos < self.view.rows {
            // If end of row, set to start of next row
            if self.col_pos == self.view.cols - 1 {
                self.row_pos += 1usize;
                self.col_pos = 0usize;
            } else {
                self.col_pos += 1usize;
            }

            Some(&self.view.mat[[raw_row_idx, raw_col_idx]])
        } else {
            None
        }
    }
}

/// Indexes matrix.
///
/// Takes row index first then column.
impl<'a, T> Index<[usize; 2]> for MatrixView<'a, T> {
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
impl<'a, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        (&self) * (&f)
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'b T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: &T) -> Matrix<T> {
        (&self) * f
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + Mul<T, Output = T>> Mul<T> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, f: T) -> Matrix<T> {
        self * (&f)
    }
}

/// Multiplies matrix by scalar.
impl<'a, 'b, 'c, T: Copy + One + Zero + Mul<T, Output = T>> Mul<&'c T> for &'b MatrixView<'a, T> {
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
impl<'a, T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<Matrix<T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<Matrix<T>> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: Matrix<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'b Matrix<T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &Matrix<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix by matrix.
impl<'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'c Matrix<T>> for &'b MatrixView<'a, T> {
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
impl<'a, 'b, T: Copy + Zero + One + Mul<T, Output = T> + Add<T, Output = T>> Mul<MatrixView<'b, T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: MatrixView<T>) -> Matrix<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<MatrixView<'b, T>> for &'c MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: MatrixView<T>) -> Matrix<T> {
        self * (&m)
    }
}

/// Multiplies matrix by matrix.
impl <'a, 'b, 'c, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'c MatrixView<'b, T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &MatrixView<T>) -> Matrix<T> {
        (&self) * m
    }
}

/// Multiplies matrix by matrix.
impl<'a, 'b, 'c, 'd, T: Copy + Zero + One + Mul<T, Output=T> + Add<T, Output=T>> Mul<&'d MatrixView<'b, T>> for &'c MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn mul(self, m: &MatrixView<T>) -> Matrix<T> {
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
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<T> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: T) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds scalar to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &T) -> Matrix<T> {
        (&self) + f
    }
}

impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c T> for &'b MatrixView<'a, T> {
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
impl<'a, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<Matrix<T>> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: Matrix<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<&'b Matrix<T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c Matrix<T>> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = self.iter().zip(m.data().iter()).map(|(u, v)| *u + *v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Add<T, Output = T>> Add<MatrixView<'b, T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: MatrixView<T>) -> Matrix<T> {
        (&self) + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<MatrixView<'b, T>> for &'c MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: MatrixView<T>) -> Matrix<T> {
        self + (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Add<T, Output = T>> Add<&'c MatrixView<'b, T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, f: &MatrixView<T>) -> Matrix<T> {
        (&self) + f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, 'd, T: Copy + One + Zero + Add<T, Output = T>> Add<&'d MatrixView<'b, T>> for &'c MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn add(self, m: &MatrixView<T>) -> Matrix<T> {
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
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'a T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &T) -> Matrix<T> {
        (&self) - f
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<T> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: T) -> Matrix<T> {
        self - (&f)
    }
}

/// Subtracts scalar from matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c T> for &'b MatrixView<'a, T> {
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

/// Adds matrix to matrix.
impl<'a, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<Matrix<T>> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: Matrix<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'b Matrix<T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &Matrix<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c Matrix<T>> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = self.iter().zip(m.data().iter()).map(|(u, v)| *u - *v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, T: Copy + One + Zero + Sub<T, Output = T>> Sub<MatrixView<'b, T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: MatrixView<T>) -> Matrix<T> {
        (&self) - (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<MatrixView<'b, T>> for &'c MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: MatrixView<T>) -> Matrix<T> {
        self - (&f)
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'c MatrixView<'b, T>> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, f: &MatrixView<T>) -> Matrix<T> {
        (&self) - f
    }
}

/// Adds matrix to matrix.
impl<'a, 'b, 'c, 'd, T: Copy + One + Zero + Sub<T, Output = T>> Sub<&'d MatrixView<'b, T>> for &'c MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn sub(self, m: &MatrixView<T>) -> Matrix<T> {
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
impl<'a, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        (&self) / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<T> for &'b MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: T) -> Matrix<T> {
        self / (&f)
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'b T> for MatrixView<'a, T> {
    type Output = Matrix<T>;

    fn div(self, f: &T) -> Matrix<T> {
        (&self) / f
    }
}

/// Divides matrix by scalar.
impl<'a, 'b, 'c, T: Copy + One + Zero + PartialEq + Div<T, Output = T>> Div<&'c T> for &'b MatrixView<'a, T> {
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
impl<'a, T: Neg<Output = T> + Copy> Neg for MatrixView<'a, T> {
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
impl<'a, 'b, T: Neg<Output = T> + Copy> Neg for &'b MatrixView<'a, T> {
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
	use super::MatrixView;
	use super::super::Matrix;

	#[test]
	#[should_panic]
	fn make_view() {
		let a = Matrix::new(3,3, vec![2.0; 9]);
		let _ = MatrixView::from_matrix(&a, [1,1], 3, 2);
	}

	#[test]
	fn add_view() {
		let a = 3.0;
		let b = Matrix::new(3,3, vec![2.0; 9]);
		let c = Matrix::new(2,2, vec![1.0; 4]);

		let d = MatrixView::from_matrix(&b, [1,1], 2, 2);

		let m_1 = &d + a;
		assert_eq!(m_1.into_vec(), vec![5.0; 4]);

		let m_2 = &d + c;
		assert_eq!(m_2.into_vec(), vec![3.0; 4]);

		let m_3 = &d + &d;
		assert_eq!(m_3.into_vec(), vec![4.0; 4]);		
	}

	#[test]
	fn sub_view() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let c = Matrix::new(3,3, vec![2.0; 9]);

		let d = MatrixView::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d - a;
		assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

		let m_2 = &d - b;
		assert_eq!(m_2.into_vec(), vec![1.0; 4]);

		let m_3 = &d - &d;
		assert_eq!(m_3.into_vec(), vec![0.0; 4]);
	}

	#[test]
	fn mul_view() {
		let a = 3.0;
		let b = Matrix::new(2,2, vec![1.0; 4]);
		let c = Matrix::new(3,3, vec![2.0; 9]);

		let d= MatrixView::from_matrix(&c, [1,1], 2, 2);

		let m_1 = &d * a;
		assert_eq!(m_1.into_vec(), vec![6.0; 4]);

		let m_2 = &d * b;
		assert_eq!(m_2.into_vec(), vec![4.0; 4]);

		let m_3 = &d * d;
		assert_eq!(m_3.into_vec(), vec![8.0; 4]);
	}

	#[test]
	fn div_view() {
		let a = 3.0;

		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixView::from_matrix(&b, [1,1], 2, 2);

		let m = c / a;
		assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
	}

	#[test]
	fn neg_view() {
		let b = Matrix::new(3,3, vec![2.0; 9]);

		let c = MatrixView::from_matrix(&b, [1,1], 2, 2);

		let m = -c;
		assert_eq!(m.into_vec(), vec![-2.0;4]);
	}

	#[test]
	fn index_view() {
		let b = Matrix::new(3,3, (0..9).collect());

		let c = MatrixView::from_matrix(&b, [1,1], 2, 2);
		
		assert_eq!(c[[0,0]], 4);
		assert_eq!(c[[0,1]], 5);
		assert_eq!(c[[1,0]], 7);
		assert_eq!(c[[1,1]], 8);
	}
}