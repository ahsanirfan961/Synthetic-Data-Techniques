# !/bin/bash

start=$(date +%s)

# Detect the number of NVIDIA GPUs and create a device string
gpu_count=$(nvidia-smi -L | wc -l)
if [ $gpu_count -eq 0 ]; then
    echo "No NVIDIA GPUs detected. Exiting."
    exit 1
fi
# Construct the CUDA device string
cuda_devices=""
for ((i=0; i<gpu_count; i++)); do
    if [ $i -gt 0 ]; then
        cuda_devices+=","
    fi
    cuda_devices+="$i"
done

# Install dependencies
apt update
apt install -y screen vim git-lfs
screen

pip install -r requirements.txt

# Check if HUGGINGFACE_TOKEN is set and log in to Hugging Face
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "HUGGINGFACE_TOKEN is defined. Logging in..."
    huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
fi

python Synthetic-Data-Techniques/create_config.py \
  --hf_token="$HUGGINGFACE_TOKEN" \
  --input_dataset_path="$INPUT_DATASET_PATH" \
  --output_dataset_path="$OUTPUT_DATASET_PATH" \
  --input_batch_size="$INPUT_BATCH_SIZE" \
  --new_max_tokens="$NEW_MAX_TOKENS" \
  --temperature="$TEMPERATURE" \
  --instruct_model_path="$INSTRUCT_MODEL_PATH" \
  --response_model_path="$RESPONSE_MODEL_PATH"


cat Synthetic-Data-Techniques/test.yaml


# if [ "$DEBUG" == "False" ]; then
#     runpodctl remove pod $RUNPOD_POD_ID
# fi

# sleep infinity