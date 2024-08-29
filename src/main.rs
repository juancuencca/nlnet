use std::vec;

use nlnet::tree::DecisionTree;

fn main() {
    let x_train = vec![
        vec![0.5, 0.3],
        vec![0.3, 0.2],
        vec![0.1, 0.8],
        vec![0.3, 0.3],
        vec![0.3, 0.2]
    ];

    let y_train: Vec<u32> = vec![
        0, 1, 1, 0, 1
    ];

    let mut classifier = DecisionTree::new(2, 2).unwrap();
    classifier.fit(&x_train, &y_train);
    classifier.print();
}
