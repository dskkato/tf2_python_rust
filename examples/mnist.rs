use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::result::Result;
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

use image::io::Reader as ImageReader;
use image::GenericImageView;

fn main() -> Result<(), Box<dyn Error>> {
    let filename = "examples/mnist/model.pb";
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python examples/mnist/mnist.py' to generate {} \
                     and try again.",
                    filename
                ),
            )
            .unwrap(),
        ));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[28, 28]);
    let img = ImageReader::open("examples/mnist/sample.png")?.decode()?;
    for (i, (_, _, pixel)) in img.pixels().enumerate() {
        x[i] = pixel.0[0] as f32 / 255.0f32;
    }

    // Load the computation graph defined by addition.py.
    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open(filename)?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let session = Session::new(&SessionOptions::new(), &graph)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(&graph.operation_by_name_required("x")?, 0, &x);
    let output = args.request_fetch(&graph.operation_by_name_required("output/Softmax")?, 0);
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
