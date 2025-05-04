#!/usr/bin/env python3
# Export Sonic UNet model to ONNX and convert to TensorRT

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import time
import sys

# Add paths
sonic_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(sonic_path)

# Import only add_ip_adapters, not the model class
from src.models.base.unet_spatio_temporal_condition import add_ip_adapters


def parse_args():
    parser = argparse.ArgumentParser(description="Export Sonic UNet model to ONNX and TensorRT")
    parser.add_argument("--unet_path", type=str, required=True, help="Path to UNet model (.pth or .pt file)")
    parser.add_argument("--output_dir", type=str, default="./tensorrt_exports", help="Output directory for ONNX and TensorRT models")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default="fp16", help="Precision for TensorRT engine")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the model")
    parser.add_argument("--height", type=int, default=64, help="Height of the input latent")
    parser.add_argument("--width", type=int, default=64, help="Width of the input latent")
    parser.add_argument("--sequence_length", type=int, default=1, help="Sequence length")
    parser.add_argument("--workspace", type=int, default=4096, help="Workspace size in MB for TensorRT")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--ip_audio_scale", type=float, default=2.5, help="Scale for IP audio adapter")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def export_onnx(model, args):
    """Export the UNet model to ONNX format"""
    os.makedirs(args.output_dir, exist_ok=True)

    onnx_path = os.path.join(args.output_dir, f"sonic_unet_{args.precision}.onnx")

    print(f"Exporting model to ONNX: {onnx_path}")

    # Create dummy inputs
    sample = torch.randn(args.batch_size, 4, args.sequence_length, args.height, args.width,
                         device="cuda", dtype=torch.float16 if args.precision == "fp16" else torch.float32)
    timestep = torch.tensor([0], device="cuda")
    encoder_hidden_states = torch.randn(args.batch_size, 77, 768,
                                        device="cuda", dtype=torch.float16 if args.precision == "fp16" else torch.float32)

    # You may need to add other inputs specific to your Sonic model here
    # These are the standard inputs for a UNet model in diffusion

    # Export to ONNX
    torch.onnx.export(
        model,
        (sample, timestep, encoder_hidden_states, None),  # Input tuple - modify based on your model's inputs
        onnx_path,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["output"],
        dynamic_axes={
            "sample": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    print(f"ONNX export completed: {onnx_path}")
    return onnx_path


def generate_trtexec_command(onnx_path, args):
    """Generate the trtexec command to convert ONNX to TensorRT"""
    engine_path = os.path.join(args.output_dir, f"sonic_unet_{args.precision}.engine")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace}",
    ]

    # Add precision flags
    if args.precision == "fp16":
        cmd.append("--fp16")

    # Add verbose flag if requested
    if args.verbose:
        cmd.append("--verbose")

    # Show device information
    cmd.append("--device=0")

    # Return both the command and the engine path
    return " ".join(cmd), engine_path


def load_unet_model(args):
    """Load the UNet model from the specified path"""
    print(f"Loading UNet model from: {args.unet_path}")

    # Load the model directly - don't try to create it from scratch
    # This assumes the .pth file contains the full model, not just weights
    try:
        unet = torch.load(args.unet_path, map_location="cpu")
        print(f"Loaded model type: {type(unet).__name__}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

        # Try loading as state dict
        print("Attempting to load as state dict...")
        state_dict = torch.load(args.unet_path, map_location="cpu")

        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # We need to know what model class to instantiate here
        print("ERROR: Cannot instantiate model from state dict without knowing model class.")
        print("Please modify this script to use your specific model class.")
        sys.exit(1)

    # Add IP adapters if needed
    add_ip_adapters(unet, [32], [args.ip_audio_scale])

    # Move to device and set to eval mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    unet.eval()

    # Set precision
    if args.precision == "fp16":
        unet.half()

    return unet


def main():
    args = parse_args()

    # Load the UNet model
    model = load_unet_model(args)

    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)

    # Export to ONNX
    onnx_path = export_onnx(model, args)

    # Generate trtexec command
    trtexec_cmd, engine_path = generate_trtexec_command(onnx_path, args)

    # Print trtexec command for the user to execute
    print("\n=== TensorRT Conversion Command ===")
    print(trtexec_cmd)
    print("\nRun this command to convert the ONNX model to TensorRT engine.")
    print(f"The resulting engine will be saved to: {engine_path}")

    # Ask if user wants to execute the command now
    response = input("\nDo you want to execute this command now? (y/n): ")
    if response.lower() == 'y':
        print("\nConverting ONNX to TensorRT...\n")
        os.system(trtexec_cmd)

        if os.path.exists(engine_path):
            print(f"\nTensorRT engine successfully generated: {engine_path}")

            # Calculate engine file size
            engine_size = os.path.getsize(engine_path) / (1024 * 1024)  # Size in MB
            print(f"Engine size: {engine_size:.2f} MB")
        else:
            print(f"\nError: TensorRT engine was not generated at {engine_path}")
    else:
        print("\nYou can run the command later to generate the TensorRT engine.")


if __name__ == "__main__":
    main()

# usage: python generate_tensorrt_engine.py --unet_path /path/to/your/unet.pth --precision fp16