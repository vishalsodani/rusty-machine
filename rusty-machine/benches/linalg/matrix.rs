use rm::linalg::matrix::Matrix;
use test::Bencher;

#[bench]
fn empty(b: &mut Bencher) {
    b.iter(|| 1)
}

#[bench]
fn mat_ref_add_100_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| {
    	&a + &c
    })
}

#[bench]
fn mat_create_add_100_100(b: &mut Bencher) {
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| {
    	let a = Matrix::new(100, 100, vec![2.0;10000]);
    	a + &c
    })
}

#[bench]
fn mat_create_100_100(b: &mut Bencher) {
    b.iter(|| {
    	let a = Matrix::new(100, 100, vec![2.0;10000]);
    	a
    })
}

#[bench]
fn mat_mul_10_10(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2.0;100]);
    let c = Matrix::new(10, 10, vec![3.0;100]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_paramul_10_10(b: &mut Bencher) {

    let a = Matrix::new(10, 10, vec![2.0;100]);
    let c = Matrix::new(10, 10, vec![3.0;100]);

    b.iter(|| a.paramul(&c))
}

#[bench]
fn mat_mul_100_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_paramul_100_100(b: &mut Bencher) {

    let a = Matrix::new(100, 100, vec![2.0;10000]);
    let c = Matrix::new(100, 100, vec![3.0;10000]);

    b.iter(|| a.paramul(&c))
}

#[bench]
fn mat_mul_128_128(b: &mut Bencher) {

    let a = Matrix::new(128, 128, vec![2.0;16384]);
    let c = Matrix::new(128, 128, vec![3.0;16384]);

    b.iter(|| &a * &c)
}

#[bench]
fn mat_paramul_128_128(b: &mut Bencher) {

    let a = Matrix::new(128, 128, vec![2.0;16384]);
    let c = Matrix::new(128, 128, vec![3.0;16384]);

    b.iter(|| a.paramul(&c))
}