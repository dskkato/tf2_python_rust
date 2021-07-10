# How to know output nodes

Use `saved_model_cli` to search required node names.

## How to know inputs/outputs names

use `saved_model_cli`

```sh
saved_model_cli show --dir ./examples/regression_savedmodel/
The given SavedModel contains the following tag-sets:
'serve'
```

```sh
saved_model_cli show --dir ./examples/regression_savedmodel/ --tag serve
The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:    SignatureDef key: "__saved_model_init_op"SignatureDef key: "train"
SignatureDef key: "weights"  
```

```sh
saved_model_cli show --dir ./examples/regression_savedmodel/ --tag serve --signature_def train
The given SavedModel SignatureDef contains the following input(s):
  inputs['x'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1)
      name: train_x:0
  inputs['y'] tensor_info:
      dtype: DT_FLOAT
      shape: (-1)
      name: train_y:0
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
      dtype: DT_FLOAT
      shape: ()
      name: StatefulPartitionedCall:0
Method name is: tensorflow/serving/predict
```

```sh
saved_model_cli show --dir ./examples/regression_savedmodel/ --tag serve --signature_def weights
The given SavedModel SignatureDef contains the following input(s):
The given SavedModel SignatureDef contains the following output(s):
  outputs['output_0'] tensor_info:
      dtype: DT_FLOAT
      shape: (1)
      name: StatefulPartitionedCall_1:0
  outputs['output_1'] tensor_info:
      dtype: DT_FLOAT
      shape: (1)
      name: StatefulPartitionedCall_1:1
Method name is: tensorflow/serving/predict
```
