# copy-of-slangpy-neuralnetwork

This is just a copy of https://github.com/shader-slang/slangpy/tree/main/experiments/neuralnetwork
The difference is that `main.py` has additional functions to save and load the model weights, and do training and inference separately.
Feel free to use the code in main.py to get you started.

Please refer to https://github.com/shader-slang/slangpy/tree/main for more information about SlangPy, including documentation and other examples.

## Scripts
Here are some scripts for training and inference with the neural texture generator.

```bash
python main.py --mode train --max_epochs 100 --save_path my_model.npz
python main.py --mode inference --save_path my_model.npz --output my_result.png
```

You should observe `my_result.png` to be a blurred version of `bernie.jpg`.