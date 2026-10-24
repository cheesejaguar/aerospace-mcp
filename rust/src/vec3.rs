//! Minimal 3-vector helpers shared by the orbital mechanics modules.

pub type Vec3 = [f64; 3];

/// Euclidean norm, summed in the same order as the original Python helper.
pub fn norm(v: Vec3) -> f64 {
    (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt()
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(v: Vec3, s: f64) -> Vec3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Unit vector in the direction of `v` (zero vector if `v` is zero).
pub fn normalize(v: Vec3) -> Vec3 {
    let n = norm(v);
    if n == 0.0 {
        [0.0; 3]
    } else {
        scale(v, 1.0 / n)
    }
}
