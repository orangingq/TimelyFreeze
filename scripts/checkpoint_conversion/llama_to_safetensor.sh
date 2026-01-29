#!/bin/bash
# convert_llama_original_to_safetensor.sh
# Usage: ./convert_llama_original_to_safetensor.sh [Llama-3.2-1B] [/path/to/original/dir]

set -e

MODEL="${1:-Llama-3.2-1B}"
ORIGINAL_DIR="${/path/to/original/dir}"
OUTPUT_DIR="${ORIGINAL_DIR}_hf"
SAFETENSORS_DIR="${OUTPUT_DIR}/safetensors"

if [ -z "$ORIGINAL_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <model_name> <original_dir>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$SAFETENSORS_DIR"

echo "[1/3] Converting Meta LLaMA original checkpoint → Hugging Face format..."
python3 scripts/checkpoint_conversion/convert_llama_to_hf.py \
  --input_dir "$ORIGINAL_DIR" \
  --model_size "${MODEL##*-}" \
  --output_dir "$OUTPUT_DIR" \
  --safe_serialization
  --llama_version "${MODEL%%-*}" \
  --num_shards 8 

echo "[2/3] Loading HF model and saving as safetensors..."
python3 - << EOF
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("$OUTPUT_DIR", device_map="cpu")
model.save_pretrained("$SAFETENSORS_DIR", safe_serialization=True)
EOF

echo "[3/3] ✅ Conversion complete!"
echo "Hugging Face checkpoint: $OUTPUT_DIR"
echo "Safetensors checkpoint: $SAFETENSORS_DIR"
