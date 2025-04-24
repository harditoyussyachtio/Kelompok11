use ndarray::{Array2, Axis};
use rand::{thread_rng, Rng};
use std::collections::HashMap;
use std::ffi::{CString, CStr};
use std::os::raw::c_char;
use csv::ReaderBuilder;
use std::fs::File;
use plotters::prelude::*;

#[no_mangle]
pub extern "C" fn train_model(path: *const c_char) -> *mut c_char {
    let c_str = unsafe { CStr::from_ptr(path) };
    let filename = c_str.to_str().unwrap_or("WaterQualityTesting.csv");

    let (inputs, targets) = match load_data(filename) {
        Ok(data) => data,
        Err(e) => return CString::new(format!("Error: {}", e)).unwrap().into_raw(),
    };

    let result = match train_neural_network(&inputs, &targets) {
        Ok(accuracy) => format!("Training complete! Accuracy: {:.2}%", accuracy),
        Err(e) => format!("Training failed: {}", e),
    };

    CString::new(result).unwrap().into_raw()
}

fn load_data(filename: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut input_data = vec![];
    let mut labels = vec![];

    for (row_index, result) in rdr.records().enumerate() {
        let record = result?;
        let mut row = vec![];

        // Ambil semua kolom kecuali yang terakhir (asumsi terakhir adalah label)
        for i in 0..record.len() - 1 {
            let raw = record.get(i).unwrap_or("0").trim().replace(",", ".");
            let val: f64 = match raw.parse() {
                Ok(v) => v,
                Err(_) => {
                    return Err(format!("Invalid float at row {}, column {}: '{}'", row_index + 1, i + 1, raw).into());
                }
            };
            row.push(val / 100.0); // normalisasi, opsional
        }

        // Ambil label (kolom terakhir)
        let label_raw = record.get(record.len() - 1).unwrap_or("0").trim();
        let label: usize = match label_raw.parse() {
            Ok(v) => v,
            Err(_) => return Err(format!("Invalid label at row {}: '{}'", row_index + 1, label_raw).into()),
        };

        input_data.push(row);
        labels.push(label);
    }

    let num_samples = input_data.len();
    let num_features = input_data[0].len();
    let num_classes = *labels.iter().max().unwrap_or(&0) + 1;

    let inputs = Array2::from_shape_vec(
        (num_samples, num_features),
        input_data.into_iter().flatten().collect(),
    )?;

    let mut targets = Array2::<f64>::zeros((num_samples, num_classes));
    for (i, &label) in labels.iter().enumerate() {
        targets[[i, label]] = 1.0;
    }

    Ok((inputs, targets))
}

fn train_neural_network(inputs: &Array2<f64>, targets: &Array2<f64>) -> Result<f64, Box<dyn std::error::Error>> {
    let input_size = inputs.shape()[1];
    let hidden_size = 16;
    let output_size = targets.shape()[1];
    let learning_rate = 0.1;
    let epochs = 1000;

    let mut nn = NeuralNetwork::new(input_size, hidden_size, output_size, learning_rate);
    nn.train(inputs, targets, epochs);

    save_training_plot("output.png").ok();

    let predictions = nn.predict(inputs)?;

    let correct = predictions
        .outer_iter()
        .zip(targets.outer_iter())
        .filter(|(p, t)| {
            let pred_class = p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i);
            let true_class = t.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i);
            pred_class == true_class
        })
        .count();

    Ok(correct as f64 / predictions.len() as f64 * 100.0)
}

struct NeuralNetwork {
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    learning_rate: f64,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, learning_rate: f64) -> Self {
        let mut rng = thread_rng();
        Self {
            weights_input_hidden: Array2::from_shape_fn((input_size, hidden_size), |_| rng.gen_range(-0.5..0.5)),
            weights_hidden_output: Array2::from_shape_fn((hidden_size, output_size), |_| rng.gen_range(-0.5..0.5)),
            learning_rate,
        }
    }

    fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, epochs: usize) {
        for _ in 0..epochs {
            let hidden = sigmoid(&inputs.dot(&self.weights_input_hidden));
            let outputs = sigmoid(&hidden.dot(&self.weights_hidden_output));

            let output_error = targets - &outputs;
            let output_delta = &output_error * &sigmoid_derivative(&outputs);

            let hidden_error = output_delta.dot(&self.weights_hidden_output.t());
            let hidden_delta = &hidden_error * &sigmoid_derivative(&hidden);

            self.weights_hidden_output += &hidden.t().dot(&output_delta).mapv(|x| x * self.learning_rate);
            self.weights_input_hidden += &inputs.t().dot(&hidden_delta).mapv(|x| x * self.learning_rate);
        }
    }

    fn predict(&self, inputs: &Array2<f64>) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let hidden = sigmoid(&inputs.dot(&self.weights_input_hidden));
        Ok(sigmoid(&hidden.dot(&self.weights_hidden_output)))
    }
}

fn sigmoid(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn sigmoid_derivative(x: &Array2<f64>) -> Array2<f64> {
    x * &(1.0 - x)
}

fn save_training_plot(filename: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Accuracy", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..1000, 0..100)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(LineSeries::new(
        (0..1000).map(|x| (x, (x as f64 / 10.0).sin().abs() as i32 * 100 / 2)),
        &RED,
    ))?;

    Ok(())
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    unsafe {
        if !s.is_null() {
            let _ = CString::from_raw(s);
        }
    }
}