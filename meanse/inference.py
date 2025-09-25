import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
import soundfile as sf
import torch
from tqdm import tqdm
from meanse.flow_model import FlowSEModel
from meanse.meanflow_model import MeanFlowSEModel
import os
import tqdm

def main(args):
    force_model_type = args.force_model_type.lower()
    device = args.device

    if force_model_type == "flow":
        model = FlowSEModel.load_from_checkpoint(args.ckpt_path, map_location=device, strict=False)
    elif force_model_type == "meanflow":
        model = MeanFlowSEModel.load_from_checkpoint(args.ckpt_path, map_location=device, strict=False)
    else:
        try:
            model = MeanFlowSEModel.load_from_checkpoint(args.ckpt_path, map_location=device, strict=True)
            print("Load MeanFlowSEModel successfully!")
        except:
            try:
                model = FlowSEModel.load_from_checkpoint(args.ckpt_path, map_location=device, strict=True)
                print("Load FlowSEModel successfully!")
            except:
                raise RuntimeError("Failed to load the model, please check the checkpoint path, or set --force_model_type to 'flow' or 'meanflow'!")

    model.eval()

    input_audios = {}
    with open(args.input_scp) as f:
        for line in f:
            utt, wav = line.strip().split()
            input_audios[utt] = wav

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir + "/wav", exist_ok=True)

    with open(args.output_dir + "/inf.scp", "w") as f:

        for uid in tqdm.tqdm(input_audios):
            wav_path = input_audios[uid]
            wav, sr = sf.read(wav_path)
            wav = torch.tensor(wav).float().to(device).view(1, -1)
            length = torch.tensor(wav.shape[-1]).to(device).view(1)

            with torch.no_grad():
                se_speech = model.enhance(wav, sr, length, args.nfe)
                se_speech = se_speech / se_speech.abs().max() * 0.9
                sf.write(args.output_dir + f"/wav/{uid}.wav", se_speech.cpu().numpy().flatten(), sr)

            print(f"{uid} {args.output_dir}/wav/{uid}.wav", file=f)

    print(f"Inference done, output dir: {args.output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_scp",
        type=str,
        required=True,
        help="Path to the tsv file containing audio samples",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./inference",
        help="Path to the output directory for writting enhanced speeches",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cuda",
        help="Device for running TorchAudio-SQUIM calculation",
    )
    parser.add_argument(
        "--force_model_type",
        type=str,
        required=False,
        default="",
        help="for the inference to load SEModel, FlowSEModel or ganSEModel",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        required=False,
        default=10,
        help="NFEs in flow enhance"
    )

    args = parser.parse_args()
    main(args)
