#[derive(Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self, &'static str> {
        if values.len() == 0 {
            return Err("Error: Vector size cannot be 0");
        }   
    
        if rows * cols != values.len() {
            return Err("Error: Vector size should match given shape");
        }

        Ok(Self { rows, cols, values })
    }

    pub fn add(mat1: &Self, mat2: &Self) -> Result<Self, &'static str> {
        if mat1.rows != mat2.rows || mat1.cols != mat2.cols {
            return Err("Error: Missmatch shape between matrices");
        }

        let mut values: Vec<f32> = Vec::new();
        
        for i in 0..mat1.rows {
            for j in 0..mat1.cols {
                let val = mat1.get_value(i, j) + mat2.get_value(i, j);
                values.push(val);
            }
        }

        Ok(Self { rows: mat1.rows, cols: mat1.cols, values})
    }

    pub fn multiply(mat1: &Self, mat2: &Self) -> Result<Self, &'static str> {
        if mat1.cols != mat2.rows {
            return Err("Error: Missmatch shape for multiplication between matrices");
        }

        let mut values: Vec<f32> = Vec::new();
        
        for i in 0..mat1.rows {
            for j in 0..mat2.cols {
                let mut sum: f32 = 0.0;
                for k in 0..mat1.cols {
                    sum += mat1.get_value(i, k) * mat2.get_value(k, j); 
                }
                values.push(sum);
            }
        }

        Ok(Self { rows: mat1.rows, cols: mat2.cols, values})
    }

    pub fn dot(&self, x: f32) -> Self {
        let mut values: Vec<f32> = Vec::new();

        for i in 0..self.rows {
            for j in 0..self.cols {
                values.push(self.get_value(i, j) * x);
            }
        }

        Self { rows: self.rows, cols: self.cols, values }
    }

    fn get_value(&self, row: usize, col: usize) -> f32 {
        self.values[(row * self.cols) + col]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_success() {
        let matrix = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(matrix.is_ok());

        assert_eq!(
            matrix.unwrap(), 
            Matrix {
                rows: 2, 
                cols: 2, 
                values: vec![1.0, 2.0, 3.0, 4.0]
            }
        );
    }

    #[test]
    fn test_matrix_creation_error_empty_vector() {
        let matrix = Matrix::new(2, 2, vec![]);
        assert!(matrix.is_err());
        assert_eq!(
            matrix.unwrap_err(), 
            "Error: Vector size cannot be 0"
        );
    }

    #[test]
    fn test_matrix_creation_error_mismatch_shape() {
        let matrix = Matrix::new(2, 2, vec![1.0, 1.0]);
        assert!(matrix.is_err());
        assert_eq!(
            matrix.unwrap_err(), 
            "Error: Vector size should match given shape"
        );
    }

    #[test]
    fn test_matrix_addition_success() {
        let m = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 2.0]).unwrap();
        let t = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 2.0]).unwrap();

        let matrix = Matrix::add(&m, &t);
        assert!(matrix.is_ok());

        assert_eq!(
            matrix.unwrap(), 
            Matrix { rows: 2, cols: 2, values: vec![2.0, 2.0, 2.0, 4.0] }
        );
    }

    #[test]
    fn test_matrix_addition_error_mismatch_shape() {
        let m = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 2.0 ]).unwrap();
        let t = Matrix::new(2, 1, vec![1.0, 1.0]).unwrap();

        let matrix = Matrix::add(&m, &t);
        assert!(matrix.is_err());

        assert_eq!(
            matrix.unwrap_err(),
            "Error: Missmatch shape between matrices"
        );
    } 
    
    #[test]
    fn test_matrix_multiplication_success() {
        let m = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 2.0]).unwrap();
        let t = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 2.0]).unwrap();

        let matrix = Matrix::multiply(&m, &t);
        assert!(matrix.is_ok());

        assert_eq!(
            matrix.unwrap(), 
            Matrix { rows: 2, cols: 2, values: vec![2.0, 3.0, 3.0, 5.0] }
        );

        let m = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 2.0]).unwrap();
        let t = Matrix::new(2, 3, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();

        let matrix = Matrix::multiply(&m, &t);
        assert!(matrix.is_ok());

        assert_eq!(
            matrix.unwrap(), 
            Matrix { rows: 2, cols: 3, values: vec![3.0, 3.0, 3.0, 5.0, 5.0, 5.0] }
        );
    }

    #[test]
    fn test_matrix_multiplication_error_mismatch_shape() {
        let m = Matrix::new(2, 1, vec![1.0, 1.0]).unwrap();
        let t = Matrix::new(2, 1, vec![1.0, 1.0]).unwrap();

        let matrix = Matrix::multiply(&m, &t);
        assert!(matrix.is_err());

        assert_eq!(
            matrix.unwrap_err(),
            "Error: Missmatch shape for multiplication between matrices"
        );
    }

    #[test] 
    fn test_matrix_dot_product_success() {
        let m = Matrix::new(1, 3, vec![1.0, 1.0, 1.0]).unwrap();
        let m = m.dot(2.0);

        assert_eq!(vec![2.0, 2.0, 2.0], m.values);
    }
}
