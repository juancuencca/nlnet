use std::collections::HashSet;

#[derive(Debug, PartialEq)]
struct Node {
    feature_index: Option<usize>,
    threshold: Option<f32>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    info_gain: Option<f32>,

    value: Option<usize>,
}

impl Node {
    fn new_decision(feature_index: usize, threshold: f32, left: Box<Node>, right: Box<Node>, info_gain: f32) -> Self {
        Self { 
            feature_index: Some(feature_index), 
            threshold: Some(threshold), 
            left: Some(left), 
            right: Some(right), 
            info_gain: Some(info_gain), 
            value: None 
        } 
    }

    fn new_leaf(value: usize) -> Self {
        Self { 
            feature_index: None, 
            threshold: None, 
            left: None, 
            right: None, 
            info_gain: None, 
            value: Some(value) 
        } 
    }
}

#[derive(Debug, PartialEq)]
struct DecisionTreeClassifier {
    root: Option<Box<Node>>,
    min_samples_split: usize,
    max_depth: usize,
}

impl DecisionTreeClassifierModel {
    fn new(min_samples_split: usize, max_depth: usize) -> Result<Self, &'static str> {
        if min_samples_split < 2 {
            return Err("Error: Minimum samples split must be greater than or equal to 2");
        }

        if max_depth < 2 {
            return Err("Error: Maximum depth must be greater than or equal to 2");
        }

        Ok(Self { root: None, min_samples_split, max_depth })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation_success() {
        let node = Box::new(Node::new_decision(0, 0.5, Box::new(Node::new_leaf(1)), Box::new(Node::new_leaf(2)), 0.33));
        assert_eq!(node, Box::new(Node {
            feature_index: Some(0), 
            threshold: Some(0.5), 
            left: Some(Box::new(Node {
                feature_index:None, 
                threshold: None, 
                left: None, 
                right: None, 
                info_gain: None, 
                value: Some(1)})), 
            right: Some(Box::new(Node {
                feature_index:None, 
                threshold: None, 
                left: None, 
                right: None, 
                info_gain: None, 
                value: Some(2)})), 
            info_gain: Some(0.33), 
            value: None
        }));
    }

    #[test]
    fn test_tree_creation_success() {
        let tree = DecisionTreeClassifierModel::new(2, 2);
        assert!(tree.is_ok());
        assert_eq!(tree.unwrap(), DecisionTreeClassifierModel {
            root: None,
            min_samples_split: 2, 
            max_depth: 2
        });
    }

    #[test]
    fn test_tree_creation_error_min_samples_split() {
        let tree = DecisionTreeClassifierModel::new(1, 2);
        assert!(tree.is_err());
        assert_eq!(tree.unwrap_err(), "Error: Minimum samples split must be greater than or equal to 2");
    }

    #[test]
    fn test_tree_creation_error_max_depth() {
        let tree = DecisionTreeClassifierModel::new(2, 1);
        assert!(tree.is_err());
        assert_eq!(tree.unwrap_err(), "Error: Maximum depth must be greater than or equal to 2");
    }
}

