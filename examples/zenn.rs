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

use ndarray;

use image::io::Reader as ImageReader;
use image::GenericImageView;

fn main() -> Result<(), Box<dyn Error>> {
    let filename = "examples/zenn/mobilenetv3large.pb";
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python examples/zenn/zenn.py' to generate {} \
                     and try again.",
                    filename
                ),
            )
            .unwrap(),
        ));
    }

    // Create input variables for our addition
    let mut x = Tensor::new(&[1, 224, 224, 3]);
    let img = ImageReader::open("examples/zenn/sample.png")?.decode()?;
    for (i, (_, _, pixel)) in img.pixels().enumerate() {
        x[3 * i] = pixel.0[0] as f32;
        x[3 * i + 1] = pixel.0[1] as f32;
        x[3 * i + 2] = pixel.0[2] as f32;
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
    let output = args.request_fetch(&graph.operation_by_name_required("Identity")?, 0);
    session.run(&mut args)?;

    // Check our results.
    let output: Tensor<f32> = args.fetch(output)?;
    let res: ndarray::Array<f32, _> = output.into();
    println!("{:?}", res);

    Ok(())
}
