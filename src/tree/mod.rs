use std::{collections::{HashMap, HashSet}, process};

pub struct BestSplit {
    pub feature_index: usize,
    pub threshold: f64,
    pub info_gain: f64,
    pub x_left: Vec<Vec<f64>>,
    pub y_left: Vec<u32>,
    pub x_right: Vec<Vec<f64>>,
    pub y_right: Vec<u32>,
}

#[derive(Debug, PartialEq)]
struct Node {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    info_gain: Option<f64>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
    value: Option<u32>,
}

impl Node {
    fn new_decision(feature_index: usize, threshold: f64, info_gain: f64, left: Box<Node>, right: Box<Node>) -> Self {
        Self { 
            feature_index: Some(feature_index), 
            threshold: Some(threshold), 
            info_gain: Some(info_gain), 
            left: Some(left), 
            right: Some(right), 
            value: None 
        } 
    }

    fn new_leaf(value: u32) -> Self {
        Self { 
            feature_index: None, 
            threshold: None, 
            info_gain: None, 
            left: None, 
            right: None, 
            value: Some(value) 
        } 
    }
}

#[derive(Debug, PartialEq)]
pub struct DecisionTree {
    root: Option<Box<Node>>,
    min_samples_split: usize,
    max_depth: usize,
}

impl DecisionTree {
    pub fn new(min_samples_split: usize, max_depth: usize) -> Result<Self, &'static str> {
        if min_samples_split < 2 { 
            return Err("Error: min_samples_split less than 2"); 
        }
        if max_depth < 2 { 
            return Err("Error: max_depth less than 2"); 
        }

        Ok(Self { root: None, min_samples_split, max_depth })
    }
}

impl DecisionTree {
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[u32]) {
        self.root = self.build_tree(x, y, 0);    
    }

    fn build_tree(&self, x: &[Vec<f64>], y: &[u32], curr_depth: usize) -> Option<Box<Node>> {
        if x.is_empty() || y.is_empty() { 
            return None; 
        }

        let num_samples = x.len();
        let num_features = x[0].len();
        
        if num_samples >= self.min_samples_split && curr_depth <= self.max_depth {
            let best_split = self.get_best_split(x, y, num_features);       

            if let Some(best_split) = best_split {
                let left_subtree = self.build_tree(&best_split.x_left, &best_split.y_left, curr_depth + 1).unwrap();
                let right_subtree = self.build_tree(&best_split.x_right, &best_split.y_right, curr_depth + 1).unwrap();
                
                return Some(Box::new(Node::new_decision(
                    best_split.feature_index, 
                    best_split.threshold, 
                    best_split.info_gain, 
                    left_subtree, 
                    right_subtree
                )));
            }
        }

        self.calculate_leaf_node(y)
    }

    fn get_best_split(&self, x: &[Vec<f64>], y: &[u32], num_features: usize) -> Option<BestSplit> {
        let mut max_info_gain = f64::NEG_INFINITY;
        let mut best_split: Option<BestSplit> = None;

        for feature_index in 0..num_features {
            let features_values = &x[feature_index];
            for &threshold in features_values {
                let (x_left, y_left, x_right, y_right) = split(x, y, num_features, feature_index, threshold)
                    .unwrap_or_else(|err| {
                        eprintln!("{err}");
                        process::exit(0);
                    });

                if y_left.len() > 0 && y_right.len() > 0 {
                    let curr_info_gain = information_gain(y, &y_left, &y_right, "gini");
                    if curr_info_gain > max_info_gain {
                        max_info_gain = curr_info_gain;
                        best_split = Some(BestSplit {
                            feature_index,
                            threshold,
                            info_gain: curr_info_gain,
                            x_left,
                            y_left,
                            x_right,
                            y_right
                        });
                    }
                }
            }
        }
        best_split
    }

    fn print_tree(&self, tree: &Option<Box<Node>>) {
        if let Some(node) = tree {
            if let Some(value) = node.value {
                println!("{value}");
            } else {
                println!("X_{:?} <= {:?} ? {:.2}", node.feature_index.unwrap(), node.threshold.unwrap(), node.info_gain.unwrap());
                print!("%sleft: ");
                self.print_tree(&node.left);
                println!("%sright: ");
                self.print_tree(&node.right);
            }
        }
    }

    pub fn print(&self) {
        self.print_tree(&self.root);
    }

    fn calculate_leaf_node(&self, y: &[u32]) -> Option<Box<Node>> {
        match most_frequent(y) {
            Some(&value) => { Some(Box::new(Node::new_leaf(value))) },
            None => { return None; }
        }
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<u32> {
        x.iter()
            .map(|x| self.make_prediction(x, &self.root))
            .collect()
    }

    fn make_prediction(&self, x: &[f64], tree: &Option<Box<Node>>) -> u32 {
        match tree {
            Some(node) => {
                if let Some(value) = node.value {
                    return value;
                }

                let feature_val = x[node.feature_index.unwrap()];
                if feature_val <= node.threshold.unwrap() {
                    self.make_prediction(x, &node.left)
                } else {
                    self.make_prediction(x, &node.right)
                }
            }
            None => panic!("Tree node is None"), // Handle this case according to your needs
        }
    }
}

pub fn accuracy(y_true: &[u32], y_pred: &[u32]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "Length of true labels and predicted labels must be equal");

    let correct_predictions = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(true_val, pred_val)| true_val == pred_val)
        .count();

    correct_predictions as f64 / y_true.len() as f64
}

fn most_frequent<T: Eq + std::hash::Hash + Copy>(items: &[T]) -> Option<&T> {
    let mut counts = HashMap::new();
    
    for item in items {
        counts.entry(item).and_modify(|e| *e += 1).or_insert(0);
    }

    counts.into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(item, _)| item)
}

fn split(x: &[Vec<f64>], y: &[u32], num_features: usize, feature_index: usize, threshold: f64) 
    -> Result<(Vec<Vec<f64>>, Vec<u32>, Vec<Vec<f64>>, Vec<u32>), &'static str> {
    if x.len() != y.len() {
        return Err("Error: mismatch shapes in x and y");
    }

    if x.len() != 0 && feature_index >= num_features {
        return Err("Error: feature_index out of bounds");
    }

    let mut x_left = Vec::new();
    let mut y_left = Vec::new();
    let mut x_right = Vec::new();
    let mut y_right = Vec::new();

    for i in 0..x.len() {
        if num_features != x[i].len() {
            return Err("Error: mismatch in number of features");
        }

        let row = x[i].clone();

        if row[feature_index] <= threshold {
            x_left.push(row);
            y_left.push(y[i]);
        } else {
            x_right.push(row);
            y_right.push(y[i]);
        }
    }

    Ok((x_left, y_left, x_right, y_right))
}

fn information_gain(y: &[u32], y_left: &[u32], y_right: &[u32], mode: &str) -> f64 {
    let l_weight = y_left.len() as f64 / y.len() as f64;
    let r_weight = y_right.len() as f64 / y.len() as f64;

    let info_gain = match mode {
        "gini" => gini_index(y) - (l_weight * gini_index(y_left)) - (r_weight * gini_index(y_right)),
        _ => entropy(y) - (l_weight * entropy(y_left)) - (r_weight * entropy(y_right))
    };

    info_gain
}

fn entropy(y: &[u32]) -> f64 {
    let class_labels = y.iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<u32>>();

    let mut entropy = 0.0;
    
    for label in class_labels {
        let prob = y.iter().filter(|&&item| item == label).count() as f64 / y.len() as f64;
        entropy += prob * (prob.log2())
    }
    
    entropy * -1.0
}

fn gini_index(y: &[u32]) -> f64 {
    let class_labels = y.iter()
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<u32>>();

    let mut gini = 0.0;
    
    for label in class_labels {
        let prob = y.iter().filter(|&&item| item == label).count() as f64 / y.len() as f64;
        gini += prob * prob
    }
    
    1.0 - gini
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation_success() {
        assert_eq!(
            Node::new_decision(0, 0.5, 0.3, Box::new(Node::new_leaf(1)), Box::new(Node::new_leaf(2))), 
            Node {
                feature_index: Some(0), 
                threshold: Some(0.5), 
                left: Some(Box::new(Node { feature_index:None, threshold: None, left: None, right: None, info_gain: None, value: Some(1) })), 
                right: Some(Box::new(Node { feature_index:None, threshold: None, left: None, right: None, info_gain: None, value: Some(2) })), 
                info_gain: Some(0.3), 
                value: None
            }
        );
    }

    #[test]
    fn test_tree_creation_success() {
        let tree = DecisionTree::new(2, 2);
        assert!(tree.is_ok());
        assert_eq!(tree.unwrap(), DecisionTree { root: None, min_samples_split: 2, max_depth: 2 });
    }

    #[test]
    fn test_tree_creation_error_min_samples_split() {
        let tree = DecisionTree::new(1, 2);
        assert!(tree.is_err());
        assert_eq!(tree.unwrap_err(), "Error: min_samples_split less than 2");
    }

    #[test]
    fn test_tree_creation_error_max_depth() {
        let tree = DecisionTree::new(2, 1);
        assert!(tree.is_err());
        assert_eq!(tree.unwrap_err(), "Error: max_depth less than 2");
    }

    #[test]
    fn test_most_frequent_success() {
        assert_eq!(most_frequent(&[1, 2, 1, 4, 3, 2, 2]), Some(&2));
        assert_eq!(most_frequent(&['a', 'b', 'a']), Some(&'a'));
        assert_eq!(most_frequent(&Vec::<u32>::new()), None);
    }

    #[test]
    fn test_split_data_success() { 
        assert_eq!(
            split(&[], &[], 0, 0, 0.5), 
            Ok((vec![], vec![], vec![], vec![]))
        );
        assert_eq!(
            split(&[vec![0.2, 0.3], vec![0.7, 0.3]], &[2, 3], 2, 0, 0.5), 
            Ok((vec![vec![0.2, 0.3]], vec![2], vec![vec![0.7, 0.3]], vec![3]))
        );
        assert_eq!(
            split(&[vec![0.2, 0.3]], &[2], 2, 0, 0.5), 
            Ok((vec![vec![0.2, 0.3]], vec![2], vec![], vec![]))
        );
    }

    #[test]
    fn test_split_data_error_mismatch_shapes() {
        assert_eq!(
            split(&[vec![0.4]], &[2, 3], 0, 0, 0.5), 
            Err("Error: mismatch shapes in x and y")
        );
    }

    #[test]
    fn test_split_data_error_feature_index() {
        assert_eq!(
            split(&[vec![0.4]], &[2], 1, 1, 0.5), 
            Err("Error: feature_index out of bounds")
        );
    }

    #[test]
    fn test_split_data_error_mismatch_num_features() {
        assert_eq!(
            split(&[vec![0.4, 0.4], vec![0.4]], &[2, 1], 2, 0, 0.5), 
            Err("Error: mismatch in number of features")
        );
    }

    #[test]
    fn test_calculate_entropy_success() {
        assert_eq!(entropy(&[1, 1, 1, 1, 2, 2, 1, 2, 1]), 0.9182958340544896);
        assert_eq!(entropy(&[1, 1, 2, 2, 1, 2]), 1.0);
        assert_eq!(entropy(&[1, 1, 1, 1, 1]), 0.0);
    }

    #[test]
    fn test_calculate_gini_index_success() {
        assert_eq!(gini_index(&[1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2]), 0.48979591836734704);
        assert_eq!(gini_index(&[1, 1, 2, 2, 1, 2]), 0.5);
        assert_eq!(gini_index(&[1, 1, 1, 1, 1]), 0.0);
    }

    #[test]
    fn test_calculate_information_gain() {
        assert_eq!(
            information_gain(&vec![1,1,1,1,1,1,1,2,2,2,2], &vec![1,1,1,2,2,2], &vec![1,1,1,1,2], "gini"),
            0.04462809917355384
        );
        assert_eq!(
            information_gain(&vec![1,1,1,1,1,1,1,2,2,2,2], &vec![1,1,1,2,2,2], &vec![1,1,1,1,2], "entropy"),
            0.07205662510638455
        );
    }
}
