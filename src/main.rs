use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use nlnet::tree::{DecisionTree, accuracy};

#[derive(Debug, Deserialize)]
struct IrisRecord {
    sepal_length: f64,
    sepal_width: f64,
    petal_length: f64,
    petal_width: f64,
    class: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path("IRIS.csv")?;

    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<u32> = Vec::new();

    for result in rdr.deserialize() {
        let record: IrisRecord = result?;

        // Map class names to 0, 1, 2
        let class_index = match record.class.as_str() {
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2,
            _ => panic!("Unknown class label"),
        };

        // Push features and label to respective vectors
        x_train.push(vec![
            record.sepal_length,
            record.sepal_width,
            record.petal_length,
            record.petal_width,
        ]);
        y_train.push(class_index);
    }

    let mut classifier = DecisionTree::new(3, 3).unwrap();
    classifier.fit(&x_train, &y_train);
    classifier.print();

    let y_pred = classifier.predict(&x_train); 
    let acc = accuracy(&y_train, &y_pred);
    
    println!("Accuracy: {acc}");
    Ok(())
}
