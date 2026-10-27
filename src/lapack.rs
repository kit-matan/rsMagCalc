//! Thin FFI to the platform LAPACK/BLAS, complex double only, for K3 (and K4/K5).
//!
//! WHY LAPACK AND NOT `faer`. The plan (RUST_KERNELS_PLAN.md §2) names faer first
//! and allows LAPACK through Accelerate "if its c64 routines prove slow at
//! 2D ≈ 3000 or its eigenvector conventions fight Colpa's ordering". The deciding
//! point was the second: the parity reference for every K3 kernel is NumPy's own
//! `cholesky` / `eigh` / `eigvals`, and on this machine NumPy links Accelerate
//! (`np.show_config()` → blas/lapack: accelerate). Inside a DEGENERATE eigenspace
//! the eigenvector basis is arbitrary, so two different eigensolvers give
//! different per-band one-magnon weights (only their sum over the degenerate group
//! is physical); calling the same library on the same matrix is the only way to
//! get the per-band split NumPy got, which keeps the band-by-band oracle tests
//! meaningful under both kernel sets. It is also zero compile time and, through
//! Accelerate's AMX path, faster than any pure-Rust eigensolver at these sizes.
//!
//! On macOS the `$NEWLAPACK$ILP64` symbols are used (LAPACK 3.9.1 interfaces,
//! 64-bit integers -- the ones NumPy ≥ 2 calls). Elsewhere the classic
//! underscore symbols from the system `lapack`/`blas` (32-bit integers) are
//! linked, untested here; the wheel story for Linux is a follow-up.
#![allow(non_snake_case, clippy::too_many_arguments)]

use num_complex::Complex64 as C;
use std::os::raw::c_char;

#[cfg(target_os = "macos")]
pub type Int = i64;
#[cfg(not(target_os = "macos"))]
pub type Int = i32;

// The hidden Fortran string-length arguments are passed explicitly (one `usize`
// per `char*`, at the end); an implementation that does not read them ignores
// the extra registers on every ABI this targets.
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    #[link_name = "zpotrf$NEWLAPACK$ILP64"]
    fn zpotrf_(uplo: *const c_char, n: *const Int, a: *mut C, lda: *const Int,
               info: *mut Int, l_uplo: usize);
    #[link_name = "zheevd$NEWLAPACK$ILP64"]
    fn zheevd_(jobz: *const c_char, uplo: *const c_char, n: *const Int, a: *mut C,
               lda: *const Int, w: *mut f64, work: *mut C, lwork: *const Int,
               rwork: *mut f64, lrwork: *const Int, iwork: *mut Int, liwork: *const Int,
               info: *mut Int, l_jobz: usize, l_uplo: usize);
    #[link_name = "zgeev$NEWLAPACK$ILP64"]
    fn zgeev_(jobvl: *const c_char, jobvr: *const c_char, n: *const Int, a: *mut C,
              lda: *const Int, w: *mut C, vl: *mut C, ldvl: *const Int, vr: *mut C,
              ldvr: *const Int, work: *mut C, lwork: *const Int, rwork: *mut f64,
              info: *mut Int, l_jobvl: usize, l_jobvr: usize);
    #[link_name = "ztrsm$NEWLAPACK$ILP64"]
    fn ztrsm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
              diag: *const c_char, m: *const Int, n: *const Int, alpha: *const C,
              a: *const C, lda: *const Int, b: *mut C, ldb: *const Int,
              l_side: usize, l_uplo: usize, l_transa: usize, l_diag: usize);
    #[link_name = "zgemm$NEWLAPACK$ILP64"]
    fn zgemm_(transa: *const c_char, transb: *const c_char, m: *const Int, n: *const Int,
              k: *const Int, alpha: *const C, a: *const C, lda: *const Int, b: *const C,
              ldb: *const Int, beta: *const C, c: *mut C, ldc: *const Int,
              l_transa: usize, l_transb: usize);
    #[link_name = "zherk$NEWLAPACK$ILP64"]
    fn zherk_(uplo: *const c_char, trans: *const c_char, n: *const Int, k: *const Int,
              alpha: *const f64, a: *const C, lda: *const Int, beta: *const f64,
              c: *mut C, ldc: *const Int, l_uplo: usize, l_trans: usize);
    #[link_name = "zpotrs$NEWLAPACK$ILP64"]
    fn zpotrs_(uplo: *const c_char, n: *const Int, nrhs: *const Int, a: *const C,
               lda: *const Int, b: *mut C, ldb: *const Int, info: *mut Int, l_uplo: usize);
    #[link_name = "ztrtri$NEWLAPACK$ILP64"]
    fn ztrtri_(uplo: *const c_char, diag: *const c_char, n: *const Int, a: *mut C,
               lda: *const Int, info: *mut Int, l_uplo: usize, l_diag: usize);
    #[link_name = "zgesv$NEWLAPACK$ILP64"]
    fn zgesv_(n: *const Int, nrhs: *const Int, a: *mut C, lda: *const Int, ipiv: *mut Int,
              b: *mut C, ldb: *const Int, info: *mut Int);
}

#[cfg(not(target_os = "macos"))]
#[link(name = "lapack")]
#[link(name = "blas")]
extern "C" {
    fn zpotrf_(uplo: *const c_char, n: *const Int, a: *mut C, lda: *const Int,
               info: *mut Int, l_uplo: usize);
    fn zheevd_(jobz: *const c_char, uplo: *const c_char, n: *const Int, a: *mut C,
               lda: *const Int, w: *mut f64, work: *mut C, lwork: *const Int,
               rwork: *mut f64, lrwork: *const Int, iwork: *mut Int, liwork: *const Int,
               info: *mut Int, l_jobz: usize, l_uplo: usize);
    fn zgeev_(jobvl: *const c_char, jobvr: *const c_char, n: *const Int, a: *mut C,
              lda: *const Int, w: *mut C, vl: *mut C, ldvl: *const Int, vr: *mut C,
              ldvr: *const Int, work: *mut C, lwork: *const Int, rwork: *mut f64,
              info: *mut Int, l_jobvl: usize, l_jobvr: usize);
    fn ztrsm_(side: *const c_char, uplo: *const c_char, transa: *const c_char,
              diag: *const c_char, m: *const Int, n: *const Int, alpha: *const C,
              a: *const C, lda: *const Int, b: *mut C, ldb: *const Int,
              l_side: usize, l_uplo: usize, l_transa: usize, l_diag: usize);
    fn zgemm_(transa: *const c_char, transb: *const c_char, m: *const Int, n: *const Int,
              k: *const Int, alpha: *const C, a: *const C, lda: *const Int, b: *const C,
              ldb: *const Int, beta: *const C, c: *mut C, ldc: *const Int,
              l_transa: usize, l_transb: usize);
    fn zherk_(uplo: *const c_char, trans: *const c_char, n: *const Int, k: *const Int,
              alpha: *const f64, a: *const C, lda: *const Int, beta: *const f64,
              c: *mut C, ldc: *const Int, l_uplo: usize, l_trans: usize);
    fn zpotrs_(uplo: *const c_char, n: *const Int, nrhs: *const Int, a: *const C,
               lda: *const Int, b: *mut C, ldb: *const Int, info: *mut Int, l_uplo: usize);
    fn ztrtri_(uplo: *const c_char, diag: *const c_char, n: *const Int, a: *mut C,
               lda: *const Int, info: *mut Int, l_uplo: usize, l_diag: usize);
    fn zgesv_(n: *const Int, nrhs: *const Int, a: *mut C, lda: *const Int, ipiv: *mut Int,
              b: *mut C, ldb: *const Int, info: *mut Int);
}

const ONE: C = C { re: 1.0, im: 0.0 };
const ZERO: C = C { re: 0.0, im: 0.0 };

/// Outcome of a Cholesky attempt: `Ok(())`, `Err(k > 0)` when the leading k x k
/// minor is not positive definite, `Err(k < 0)` for an argument error.
pub fn zpotrf_upper(n: usize, a: &mut [C]) -> Result<(), Int> {
    debug_assert!(a.len() >= n * n);
    let nn = n as Int;
    let mut info: Int = 0;
    unsafe { zpotrf_(&(b'U' as c_char), &nn, a.as_mut_ptr(), &nn, &mut info, 1) };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// Eigenvalues (ascending, into `w`) and eigenvectors (columns of `a`, column
/// major) of a Hermitian matrix given by its upper triangle.
pub fn zheevd_vectors(n: usize, a: &mut [C], w: &mut [f64]) -> Result<(), Int> {
    let nn = n as Int;
    let lwork = (2 * n + n * n).max(1);
    let lrwork = (1 + 5 * n + 2 * n * n).max(1);
    let liwork = (3 + 5 * n).max(1);
    let mut work = vec![ZERO; lwork];
    let mut rwork = vec![0.0f64; lrwork];
    let mut iwork = vec![0 as Int; liwork];
    let mut info: Int = 0;
    unsafe {
        zheevd_(&(b'V' as c_char), &(b'U' as c_char), &nn, a.as_mut_ptr(), &nn,
                w.as_mut_ptr(), work.as_mut_ptr(), &(lwork as Int), rwork.as_mut_ptr(),
                &(lrwork as Int), iwork.as_mut_ptr(), &(liwork as Int), &mut info, 1, 1)
    };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// All eigenvalues of a general complex matrix (column major; destroyed).
pub fn zgeev_values(n: usize, a: &mut [C], w: &mut [C]) -> Result<(), Int> {
    let nn = n as Int;
    let mut rwork = vec![0.0f64; (2 * n).max(1)];
    let mut info: Int = 0;
    let mut dummy = [ZERO];
    // Workspace query, then the call.
    let mut wq = [ZERO];
    let m1: Int = -1;
    unsafe {
        zgeev_(&(b'N' as c_char), &(b'N' as c_char), &nn, a.as_mut_ptr(), &nn,
               w.as_mut_ptr(), dummy.as_mut_ptr(), &1, dummy.as_mut_ptr(), &1,
               wq.as_mut_ptr(), &m1, rwork.as_mut_ptr(), &mut info, 1, 1)
    };
    if info != 0 {
        return Err(info);
    }
    let lwork = (wq[0].re as usize).max(2 * n).max(1);
    let mut work = vec![ZERO; lwork];
    unsafe {
        zgeev_(&(b'N' as c_char), &(b'N' as c_char), &nn, a.as_mut_ptr(), &nn,
               w.as_mut_ptr(), dummy.as_mut_ptr(), &1, dummy.as_mut_ptr(), &1,
               work.as_mut_ptr(), &(lwork as Int), rwork.as_mut_ptr(), &mut info, 1, 1)
    };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// B <- B U^{-1} for an upper-triangular U (n x n), B being m x n, column major
/// with leading dimension `ldb`.
pub fn ztrsm_right_upper(m: usize, n: usize, u: &[C], b: &mut [C], ldb: usize) {
    unsafe {
        ztrsm_(&(b'R' as c_char), &(b'U' as c_char), &(b'N' as c_char), &(b'N' as c_char),
               &(m as Int), &(n as Int), &ONE, u.as_ptr(), &(n as Int), b.as_mut_ptr(),
               &(ldb as Int), 1, 1, 1, 1)
    };
}

/// B <- U^{-1} B for an upper-triangular U (n x n), B being n x k, column major.
pub fn ztrsm_left_upper(n: usize, k: usize, u: &[C], b: &mut [C]) {
    unsafe {
        ztrsm_(&(b'L' as c_char), &(b'U' as c_char), &(b'N' as c_char), &(b'N' as c_char),
               &(n as Int), &(k as Int), &ONE, u.as_ptr(), &(n as Int), b.as_mut_ptr(),
               &(n as Int), 1, 1, 1, 1)
    };
}

/// C <- op(A) op(B), column major: A is m x k (lda), B is k x n (ldb), C m x n (ldc).
/// `conj_b` selects op(B) = B^H (B then stored n x k).
pub fn zgemm(m: usize, n: usize, k: usize, a: &[C], lda: usize, b: &[C], ldb: usize,
             conj_b: bool, c: &mut [C], ldc: usize) {
    let tb = if conj_b { b'C' } else { b'N' } as c_char;
    unsafe {
        zgemm_(&(b'N' as c_char), &tb, &(m as Int), &(n as Int), &(k as Int), &ONE,
               a.as_ptr(), &(lda as Int), b.as_ptr(), &(ldb as Int), &ZERO,
               c.as_mut_ptr(), &(ldc as Int), 1, 1)
    };
}

// The three dense kernels of one multifrontal front (`sparse_guard.rs`): column
// major, LOWER triangle, leading dimension `ld` (the front order, so that the
// pivot block, the off-diagonal block and the update block are views of one
// buffer).

/// Lower Cholesky of the leading n x n block, in place: `Err(k > 0)` when the
/// leading k x k minor is not positive definite -- the same test `zpotrf_upper`
/// makes, on the other triangle.
pub fn zpotrf_lower_ld(n: usize, a: &mut [C], ld: usize) -> Result<(), Int> {
    let mut info: Int = 0;
    unsafe { zpotrf_(&(b'L' as c_char), &(n as Int), a.as_mut_ptr(), &(ld as Int), &mut info, 1) };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// B <- B L^{-H} for a lower-triangular L (n x n, leading dimension `lda`), B
/// being m x n with leading dimension `ldb`.
pub fn ztrsm_right_lower_conj(m: usize, n: usize, l: *const C, lda: usize, b: *mut C, ldb: usize) {
    unsafe {
        ztrsm_(&(b'R' as c_char), &(b'L' as c_char), &(b'C' as c_char), &(b'N' as c_char),
               &(m as Int), &(n as Int), &ONE, l, &(lda as Int), b, &(ldb as Int), 1, 1, 1, 1)
    };
}

/// C <- C - A A^H on the lower triangle; A is n x k (lda), C n x n (ldc).
pub fn zherk_lower_sub(n: usize, k: usize, a: *const C, lda: usize, c: *mut C, ldc: usize) {
    let (alpha, beta) = (-1.0f64, 1.0f64);
    unsafe {
        zherk_(&(b'L' as c_char), &(b'N' as c_char), &(n as Int), &(k as Int), &alpha, a,
               &(lda as Int), &beta, c, &(ldc as Int), 1, 1)
    };
}

// K5 (`scga.rs`): eigenvalues only, a Hermitian positive-definite solve, the
// inverse's diagonal from the Cholesky factor, and the general LU solve NumPy's
// `inv` makes, for the case the matrix is not positive definite.

/// Eigenvalues only (ascending, into `w`) of a Hermitian matrix, column major,
/// read from the `uplo` triangle; `a` is destroyed. The workspace is the size
/// LAPACK asks for (a query first), as NumPy's `eigvalsh` does: the blocking of
/// the tridiagonal reduction depends on it, and with it the rounding.
pub fn zheevd_values(n: usize, a: &mut [C], w: &mut [f64], uplo: u8) -> Result<(), Int> {
    let nn = n as Int;
    let mut info: Int = 0;
    let (mut wq, mut rq, mut iq) = ([ZERO], [0.0f64], [0 as Int]);
    let m1: Int = -1;
    unsafe {
        zheevd_(&(b'N' as c_char), &(uplo as c_char), &nn, a.as_mut_ptr(), &nn,
                w.as_mut_ptr(), wq.as_mut_ptr(), &m1, rq.as_mut_ptr(), &m1,
                iq.as_mut_ptr(), &m1, &mut info, 1, 1)
    };
    if info != 0 {
        return Err(info);
    }
    let lwork = (wq[0].re as usize).max(n + 1);
    let lrwork = (rq[0] as usize).max(n.max(1));
    let liwork = (iq[0] as usize).max(1);
    let mut work = vec![ZERO; lwork];
    let mut rwork = vec![0.0f64; lrwork];
    let mut iwork = vec![0 as Int; liwork];
    unsafe {
        zheevd_(&(b'N' as c_char), &(uplo as c_char), &nn, a.as_mut_ptr(), &nn,
                w.as_mut_ptr(), work.as_mut_ptr(), &(lwork as Int), rwork.as_mut_ptr(),
                &(lrwork as Int), iwork.as_mut_ptr(), &(liwork as Int), &mut info, 1, 1)
    };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// B <- A^{-1} B from the upper Cholesky factor of `zpotrf_upper` (n x n), B being
/// n x k, column major.
pub fn zpotrs_upper(n: usize, k: usize, u: &[C], b: &mut [C]) -> Result<(), Int> {
    let mut info: Int = 0;
    unsafe {
        zpotrs_(&(b'U' as c_char), &(n as Int), &(k as Int), u.as_ptr(), &(n as Int),
                b.as_mut_ptr(), &(n as Int), &mut info, 1)
    };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// U <- U^{-1} in place for an upper-triangular U (n x n, column major).
pub fn ztrtri_upper(n: usize, u: &mut [C]) -> Result<(), Int> {
    let mut info: Int = 0;
    unsafe {
        ztrtri_(&(b'U' as c_char), &(b'N' as c_char), &(n as Int), u.as_mut_ptr(),
                &(n as Int), &mut info, 1, 1)
    };
    if info == 0 { Ok(()) } else { Err(info) }
}

/// B <- A^{-1} B by LU with partial pivoting (A n x n, B n x k, column major; A is
/// overwritten by its factors). `Err(k > 0)`: A is exactly singular.
pub fn zgesv(n: usize, k: usize, a: &mut [C], b: &mut [C]) -> Result<(), Int> {
    let mut info: Int = 0;
    let mut ipiv = vec![0 as Int; n.max(1)];
    unsafe {
        zgesv_(&(n as Int), &(k as Int), a.as_mut_ptr(), &(n as Int), ipiv.as_mut_ptr(),
               b.as_mut_ptr(), &(n as Int), &mut info)
    };
    if info == 0 { Ok(()) } else { Err(info) }
}
