use super::Matrix;

use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero, Float, FromPrimitive};

#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a, T: 'a> {
	window_corner: [usize; 2],
	rows: usize,
	cols: usize,
	mat: &'a Matrix<T>,
}

impl<'a, T:'a> MatrixView<'a, T> {

	pub fn rows(&self) -> usize {
		self.rows
	}

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
	pub fn from_matrix(mat: &'a Matrix<T>, start: [usize;2], rows: usize, cols: usize) -> MatrixView<'a, T> {
		assert!(start[0] + rows <= mat.rows(), "View dimensions exceed matrix dimensions.");
		assert!(start[1] + cols <= mat.cols(), "View dimensions exceed matrix dimensions.");

		MatrixView {
			window_corner: start,
			rows: rows,
			cols: cols,
			mat: mat,
		}
	}
}

impl<'a, T: 'a + Copy> MatrixView<'a, T> {

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
	/// let view_data = view.iter().collect::<Vec<usize>>();
	/// assert_eq!(view_data, vec![4,5,7,8]);
	/// ```
	pub fn iter(&self) -> ViewIter<'a, T> {
		ViewIter {
			view: *self,
			row_pos: 0,
			col_pos: 0,
		}
	}
}

#[derive(Debug)]
pub struct ViewIter<'a, T: 'a> {
	view: MatrixView<'a, T>,
	row_pos: usize,
	col_pos: usize,
}

impl<'a, T: 'a + Copy> Iterator for ViewIter<'a, T> {
	type Item = T;

	fn next(&mut self) -> Option<Self::Item> {
		let raw_row_idx = self.view.window_corner[0] + self.row_pos;
		let raw_col_idx = self.view.window_corner[1] + self.col_pos;

		if self.row_pos < self.view.rows-1 {
			if self.col_pos == self.view.cols - 1 {
				self.row_pos += 1usize;
				self.col_pos = 0usize;
			} else {
				self.col_pos += 1usize;
			}
		} else if self.row_pos == self.view.rows-1 {
			if self.col_pos == self.view.cols - 1 {
				self.row_pos += 1usize;
			} else {
				self.col_pos += 1usize;
			}
		} else {
			return None;
		}

		Some(self.view.mat[[raw_row_idx, raw_col_idx]])
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
        let new_data: Vec<T> = self.iter().map(|v| v * (*f)).collect();

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
                        new_data[i*m.cols() + j] = *new_data.get_unchecked(i*m.cols() + j) +
                        							*self.mat.data().get_unchecked((self.window_corner[0] + i) * self.cols + (self.window_corner[1] + k)) *
                        							*m.data().get_unchecked(k*m.cols + j);
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
        let new_data: Vec<T> = self.iter().map(|v| v + (*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}