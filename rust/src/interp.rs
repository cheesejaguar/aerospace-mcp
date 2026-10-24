//! One-dimensional linear interpolation with `numpy.interp` semantics.

/// Interpolate `fp(xp)` at `x`, clamping to the end values outside `xp`.
///
/// Mirrors `numpy.interp` for non-decreasing `xp` (the Python implementation
/// this crate replaces used it for thrust curves and atmosphere tables):
/// the bracketing index is the last `j` with `xp[j] <= x`, and the result is
/// `slope * (x - xp[j]) + fp[j]`. Unsorted input never panics; it just gives
/// an unspecified (but finite) result, as numpy does.
///
/// `xp` and `fp` must be non-empty and of equal length.
pub fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len().min(fp.len());
    interp_by(x, n, |i| xp[i], |i| fp[i])
}

/// [`interp`] over `(x, y)` samples, without splitting them into two slices.
pub fn interp_pairs(x: f64, samples: &[(f64, f64)]) -> f64 {
    interp_by(x, samples.len(), |i| samples[i].0, |i| samples[i].1)
}

fn interp_by(x: f64, n: usize, xp: impl Fn(usize) -> f64, fp: impl Fn(usize) -> f64) -> f64 {
    debug_assert!(n > 0, "interp requires at least one sample");
    if x.is_nan() {
        return x;
    }
    if n == 1 || x < xp(0) {
        return fp(0);
    }
    if x >= xp(n - 1) {
        return fp(n - 1);
    }
    // First index with xp > x (a partition point), by binary search.
    let (mut lo, mut hi) = (0, n);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if xp(mid) <= x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let j = lo.saturating_sub(1).min(n - 2);
    let (x0, x1, y0, y1) = (xp(j), xp(j + 1), fp(j), fp(j + 1));
    if x0 == x {
        return y0;
    }
    let slope = (y1 - y0) / (x1 - x0);
    let mut y = slope * (x - x0) + y0;
    if y.is_nan() {
        y = slope * (x - x1) + y1;
        if y.is_nan() && y0 == y1 {
            y = y0;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::{interp, interp_pairs};

    #[test]
    fn matches_numpy_semantics() {
        let xp = [0.0, 1.0, 3.0];
        let fp = [10.0, 20.0, 0.0];
        assert_eq!(interp(-1.0, &xp, &fp), 10.0);
        assert_eq!(interp(0.0, &xp, &fp), 10.0);
        assert_eq!(interp(0.5, &xp, &fp), 15.0);
        assert_eq!(interp(1.0, &xp, &fp), 20.0);
        assert_eq!(interp(2.0, &xp, &fp), 10.0);
        assert_eq!(interp(3.0, &xp, &fp), 0.0);
        assert_eq!(interp(4.0, &xp, &fp), 0.0);
        assert!(interp(f64::NAN, &xp, &fp).is_nan());
    }

    #[test]
    fn duplicate_knots_take_the_later_segment() {
        let xp = [0.0, 6.0, 6.0, 7.0];
        let fp = [100.0, 100.0, 0.0, 0.0];
        assert_eq!(interp(6.0, &xp, &fp), 0.0);
        assert_eq!(interp(5.9, &xp, &fp), 100.0);
    }

    #[test]
    fn unsorted_input_does_not_panic() {
        let xp = [3.0, 1.0, 2.0];
        let fp = [1.0, 2.0, 3.0];
        for x in [0.0, 1.5, 2.5, 3.5] {
            assert!(interp(x, &xp, &fp).is_finite());
        }
    }

    #[test]
    fn pairs_match_slices() {
        let xp = [0.0, 1.0, 1.0, 3.0, 7.0];
        let fp = [10.0, 20.0, 5.0, 0.0, 4.0];
        let pairs: Vec<(f64, f64)> = xp.iter().copied().zip(fp).collect();
        for k in -10..=80 {
            let x = f64::from(k) / 10.0;
            assert_eq!(
                interp(x, &xp, &fp).to_bits(),
                interp_pairs(x, &pairs).to_bits(),
                "x={x}"
            );
        }
    }
}
