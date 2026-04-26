#!/bin/bash
# End-to-end deployment: build image â†’ push to ACR â†’ trigger DLIS pipeline
set -euo pipefail

ACR_NAME="f9309c3acdd842848c88032e1ec736d2"
IMAGE_NAME="qwen3-06b-ranker"
IMAGE_TAG="${1:-v1}"
CHECKPOINT_DIR="${2:-checkpoint-1128}"

FULL_IMAGE="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

echo "=== Step 0: Verify checkpoint ==="
if [ ! -f "${CHECKPOINT_DIR}/config.json" ]; then
    echo "ERROR: ${CHECKPOINT_DIR}/config.json not found. Copy the checkpoint first:"
    echo "  cp -r /path/to/output/v9_Qwen3-0.6B_all_ep5/checkpoint-1128 ./${CHECKPOINT_DIR}/"
    exit 1
fi

echo "=== Step 1: Build Docker image ==="
# Bake checkpoint into image for simpler deployment
cp -r "${CHECKPOINT_DIR}" model_tmp
docker build -t "${FULL_IMAGE}" --build-arg MODEL_DIR=model_tmp .
rm -rf model_tmp

echo "=== Step 2: Push to ACR ==="
az acr login --name "${ACR_NAME}"
docker push "${FULL_IMAGE}"

echo "=== Step 3: Convert AML â†’ DLIS Docker ==="
echo "Trigger ADO pipeline manually:"
echo "  Pipeline: IFF-Deployment_Deploy"
echo "  Image: ${FULL_IMAGE}"
echo ""
echo "=== Step 4: DLIS Deploy ==="
echo "Update DLIS deploy pipeline with:"
echo "  Application: ${IMAGE_NAME}"
echo "  Image: dlisfalconprodcontainerregistry.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  GPU: 1 (0.6B model)"
echo ""
echo "Done! Image pushed: ${FULL_IMAGE}"
