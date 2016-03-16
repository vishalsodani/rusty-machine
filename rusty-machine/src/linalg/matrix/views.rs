use super::Matrix;

use std::ops::{Mul, Add, Div, Sub, Index, Neg};
use libnum::{One, Zero, Float, FromPrimitive};

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

	pub fn from_matrix(mat: &'a Matrix<T>, start: [usize;2], rows: usize, cols: usize) -> MatrixView<'a, T> {
		assert!(start[0] + rows < mat.rows(), "View dimensions exceed matrix dimensions.");
		assert!(start[1] + cols < mat.cols(), "View dimensions exceed matrix dimensions.");

		MatrixView {
			window_corner: start,
			rows: rows,
			cols: cols,
			mat: mat,
		}
	}
}