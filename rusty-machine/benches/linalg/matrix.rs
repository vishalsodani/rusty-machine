use rm::linalg::matrix::Matrix;
use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

#[bench]
fn mat_add(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2.0;100]);
    let c = Matrix::new(10, 10, vec![3.0;100]);

    b.iter(|| &a + &c)
}

#[bench]
fn mat_mul_10(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2.0;100]);
    let c = Matrix::new(10, 10, vec![3.0;100]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_paramul_10(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2.0;100]);
    let c = Matrix::new(10, 10, vec![3.0;100]);

    b.iter(|| a.paramul(&c))
}

#[bench]
fn mat_mul_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_paramul_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| a.paramul(&c))
}

#[bench]
fn mat_mul_1000(b: &mut Bencher) {

    let a = Matrix::new(1000, 1000, vec![2.0;1000000]);
    let c = Matrix::new(1000, 1000, vec![3.0;1000000]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_paramul_1000(b: &mut Bencher) {

    let a = Matrix::new(1000, 1000, vec![2.0;1000000]);
    let c = Matrix::new(1000, 1000, vec![3.0;1000000]);

    b.iter(|| a.paramul(&c))
}