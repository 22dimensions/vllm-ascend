# Qwen3-32B

## Introduction

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support


## Supported Features

Refer to [supported features](../user_guide/support_matrix/supported_models.md) to get the model's supported feature matrix.

Refer to [feature guide](../user_guide/feature_guide/index.md) to get the feature's configuration.

## Environment Preparation

### Model Weight

- `Qwen3-32B`(BF16 version): require 2 Atlas 800 A3 (64G × 16) nodes or 4 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/Qwen/Qwen3-32B/files)

- `Qwen3-32B-W8A8`(W8A8 Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W8A8/files)

- `Qwen3-32B-W4A4`(W4A4 Quantized version): require 1 Atlas 800 A3 (64G × 16) node or 2 Atlas 800 A2 (64G × 8) nodes. [Download model weight](https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4/files)

### Installation



### Install modelslim and Convert Model

:::{note}
You can choose to convert the model yourself or use the quantized model we uploaded,
see https://www.modelscope.cn/models/vllm-ascend/Qwen3-32B-W4A4
:::

```bash
git clone -b tag_MindStudio_8.2.RC1.B120_002 https://gitcode.com/Ascend/msit
cd msit/msmodelslim

# Install by run this script
bash install.sh
pip install accelerate
# transformers 4.51.0 is required for Qwen3 series model
# see https://gitcode.com/Ascend/msit/blob/master/msmodelslim/example/Qwen/README.md#%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE
pip install transformers==4.51.0

cd example/Qwen
# Original weight path, Replace with your local model path
MODEL_PATH=/home/models/Qwen3-32B
# Path to save converted weight, Replace with your local path
SAVE_PATH=/home/models/Qwen3-32B-w4a4

python3 w4a4.py --model_path $MODEL_PATH \
                --save_directory $SAVE_PATH \
                --calib_file ../common/qwen_qwen3_cot_w4a4.json \
                --trust_remote_code True \
                --batch_size 1
```

### Verify the Quantized Model

The converted model files look like:

```bash
.
|-- config.json
|-- configuration.json
|-- generation_config.json
|-- quant_model_description.json
|-- quant_model_weight_w4a4_flatquant_dynamic-00001-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00002-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00003-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00004-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00005-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00006-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00007-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00008-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00009-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00010-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic-00011-of-00011.safetensors
|-- quant_model_weight_w4a4_flatquant_dynamic.safetensors.index.json
|-- tokenizer.json
|-- tokenizer_config.json
`-- vocab.json
```

## Deployment

### Online Serving on Single NPU

```bash
vllm serve /home/models/Qwen3-32B-w4a4 --served-model-name "qwen3-32b-w4a4" --max-model-len 4096 --quantization ascend
```

Once your server is started, you can query the model with input prompts.

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3-32b-w4a4",
        "prompt": "what is large language model?",
        "max_tokens": "128",
        "top_p": "0.95",
        "top_k": "40",
        "temperature": "0.0"
    }'
```

### Offline Inference on Single NPU

:::{note}
To enable quantization for ascend, quantization method must be "ascend".
:::

```python

from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40)

llm = LLM(model="/home/models/Qwen3-32B-w4a4",
          max_model_len=4096,
          quantization="ascend")

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
