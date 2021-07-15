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
use tensorflow::DEFAULT_SERVING_SIGNATURE_DEF_KEY;

use ndarray;

use image::io::Reader as ImageReader;
use image::GenericImageView;

fn main() -> Result<(), Box<dyn Error>> {
    let export_dir = "examples/zenn_savedmodel"; // y = w * x + b
    if !Path::new(export_dir).exists() {
        return Err(Box::new(
            Status::new_set(
                Code::NotFound,
                &format!(
                    "Run 'python zenn_savedmodel.py' to generate \
                     {} and try again.",
                    export_dir
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

    // Load the saved model exported by zenn_savedmodel.py.
    let mut graph = Graph::new();
    let bundle =
        SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, export_dir)?;
    let session = &bundle.session;

    // get in/out operations
    let signature = bundle
        .meta_graph_def()
        .get_signature(DEFAULT_SERVING_SIGNATURE_DEF_KEY)?;
    let x_info = signature.get_input("input_1")?;
    let op_x = &graph.operation_by_name_required(&x_info.name().name)?;
    let output_info = signature.get_output("Predictions")?;
    let op_output = &graph.operation_by_name_required(&output_info.name().name)?;

    // Run the graph.
    let mut args = SessionRunArgs::new();
    args.add_feed(op_x, 0, &x);
    let token_output = args.request_fetch(op_output, 0);
    session.run(&mut args)?;

    // Check our results.
    let output: Tensor<f32> = args.fetch(token_output)?;
    let res: ndarray::Array<f32, _> = output.into();
    println!("{:?}", res);

    Ok(())
}
