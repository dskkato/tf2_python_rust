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

use image::io::Reader as ImageReader;
use image::GenericImageView;

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "examples/mnist_savedmodel"; // y = w * x + b
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python mnist_savedmodel.py' to generate \
                     {} and try again.",
                    export_dir
                ),
            )
            .unwrap(),
        ));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[28, 28]);
    let img = ImageReader::open("examples/mnist_savedmodel/sample.png")?.decode()?;
    for (i, (_, _, pixel)) in img.pixels().enumerate() {
        x[i] = pixel.0[0] as f32 / 255.0f32;
    }

    // Load the saved model exported by regression_savedmodel.py.
    let mut graph = Graph::new();
    let session =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?.session;
    let op_x = graph.operation_by_name_required("serving_default_sequential_input")?;
    let op_predict = graph.operation_by_name_required("StatefulPartitionedCall")?;

    // Train the model (e.g. for fine tuning).
    let mut args = SessionRunArgs::new();
    args.add_feed(&op_x, 0, &x);
    let output = args.request_fetch(&op_predict, 0);
    session.run(&mut args)?;

    // Check our results.
    let mut res = Vec::new();
    let output: Tensor<f32> = args.fetch(output)?;
    for i in 0..10 {
        res.push(output[i]);
    }

    println!("{:?}", res);

    Ok(())
}
