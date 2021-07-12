use random::Source;
use std::error::Error;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::SavedModelBundle;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "examples/regression_savedmodel"; // y = w * x + b
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python regression_savedmodel.py' to generate \
                     {} and try again.",
                    export_dir
                ),
            )
            .unwrap(),
        ));
    }

    // Generate some test data.
    let w = 0.1;
    let b = 0.3;
    let num_points = 100;
    let steps = 201;
    let mut rand = random::default();
    let mut x = Tensor::new(&[num_points as u64]);
    let mut y = Tensor::new(&[num_points as u64]);
    for i in 0..num_points {
        x[i] = (2.0 * rand.read::<f64>() - 1.0) as f32;
        y[i] = w * x[i] + b;
    }

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let session =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?.session;
    let op_x = graph.operation_by_name_required("train_x")?;
    let op_y = graph.operation_by_name_required("train_y")?;
    let op_train = graph.operation_by_name_required("StatefulPartitionedCall")?;
    let op_w = graph.operation_by_name_required("StatefulPartitionedCall_1")?;
    let op_b = graph.operation_by_name_required("StatefulPartitionedCall_1")?;

    // Train the model (e.g. for fine tuning).
    let mut train_step = SessionRunArgs::new();
    train_step.add_feed(&op_x, 0, &x);
    train_step.add_feed(&op_y, 0, &y);
    train_step.request_fetch(&op_train, 0);
    for _ in 0..steps {
        session.run(&mut train_step)?;
    }

    // Grab the data out of the session.
    let mut output_step = SessionRunArgs::new();
    let w_ix = output_step.request_fetch(&op_w, 0);
    let b_ix = output_step.request_fetch(&op_b, 1);
    session.run(&mut output_step)?;

    // Check our results.
    let w_hat: f32 = output_step.fetch(w_ix)?[0];
    let b_hat: f32 = output_step.fetch(b_ix)?[0];
    println!(
        "Checking w: expected {}, got {}. {}",
        w,
        w_hat,
        if (w - w_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    println!(
        "Checking b: expected {}, got {}. {}",
        b,
        b_hat,
        if (b - b_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        }
    );
    Ok(())
}
