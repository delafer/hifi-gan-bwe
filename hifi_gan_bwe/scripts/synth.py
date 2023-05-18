""" HiFi-GAN+ audio synthesis

This script runs inference using a pretrained HiFi-GAN+ model. It loads
an audio file in an format supported by the audioread package, runs the
model forward, and then writes the results to an output file, in any
format supported by the soundfile library.

"""

import argparse
from pathlib import Path

import audioread
import numpy as np
import soundfile
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio.transforms as T
import torchaudio
import librosa
from hifi_gan_bwe import models


def main() -> None:
    parser = argparse.ArgumentParser(description="HiFi-GAN+ Bandwidth Extender")
    parser.add_argument(
        "model",
        help="pretrained model name or path",
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="input audio file path",
    )
    parser.add_argument(
        "target_path",
        type=Path,
        help="output audio file path",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="torch device to use for synthesis (ex: cpu, cuda, cuda:1, etc.)",
    )
    parser.add_argument(
        "--fade_stride",
        type=float,
        default=30,
        help="streaming chunk length, in seconds",
    )
    parser.add_argument(
        "--fade_length",
        type=float,
        default=0.025,
        help="cross-fading overlap, in seconds",
    )

    args = parser.parse_args()

    # load the model
    torch.set_grad_enabled(False)
    model = models.BandwidthExtender.from_pretrained(args.model).to(args.device)

    # load the source audio file
    with audioread.audio_open(str(args.source_path)) as input_:
        sample_rate = input_.samplerate
        # sample_rate=48000
        audio = (
            np.hstack([np.frombuffer(b, dtype=np.int16) for b in input_])
            .reshape([-1, input_.channels])
            .astype(np.float32)
            / 32767.0
        )
        # audio = soundfile.read(args.source_path,dtype='float32',samplerate=48000)[0]
        # audio=torchaudio.load(args.source_path)[0].T
        # audio=librosa.load(args.source_path,sr=48000,mono=False)[0].T
        # print("audio",audio.shape)
        # audio = np.expand_dims(audio, axis=1)
    # run the bandwidth extender on each audio channel
    inputs = torch.from_numpy(audio.copy()).to(args.device)
    print(inputs.shape)
    audio = (
        torch.stack([_stream(args, model, x, sample_rate) for x in inputs.T])
        .T.cpu()
        .numpy()
    )

    # save the output file
    soundfile.write(args.target_path, audio, samplerate=int(model.sample_rate))


def _stream(
    args: argparse.Namespace,
    model: torch.nn.Module,
    x: torch.Tensor,
    sample_rate: int,
) -> torch.Tensor:
    stride_samples = int(args.fade_stride) * sample_rate
    fade_samples = int(args.fade_length * sample_rate)
    print("fade samples",fade_samples)
    print("stride samples",stride_samples)
    # create a linear cross-fader
    fade_in = torch.linspace(0, 1, fade_samples).to(x.device)
    fade_ou = fade_in.flip(0)
    window_size=stride_samples + fade_samples
    stride=stride_samples
    print("input length",len(x))
    # Calculate the number of elements to pad
    # pad_size = window_size - (len(x) - stride) % window_size
    # out=np.floor((len(x) - window_size)/stride + 1)
    # unused = len(x) - out*(window_size-1) - 1
    res=len(x)%stride_samples
    print("res",res)

    if res > 0:
        n=int(np.ceil(len(x)/stride_samples))
        print("n",n)
        pad_size = n*stride_samples +fade_samples - len(x)
        print("pad size",int(pad_size))
        padded_x=F.pad (x, (0,int(pad_size)), mode = "constant", value = 0.0)
        # padded_x = torch.cat([x, torch.zeros(int(pad_size), dtype=x.dtype, device=x.device)])
        print("pad size",pad_size)
        print("padded length",len(padded_x))
        print("input length minus padding",len(padded_x)-pad_size)
        
    else:
        pad_size=0
        print("pad size",0)

        padded_x = x
    
    # Pad the input tensor with zeros
    
    
    # window the audio into overlapping frames
    frames = padded_x.unfold(
        dimension=0,
        size=stride_samples + fade_samples,
        step=stride_samples,
    )
    print("frames",len(frames))
    print("frames shape",frames.shape)
    print("sum of frames",sum([len(frames[i]) for i in range(len(frames))]))
    # exit()
    prev = torch.zeros_like(fade_ou)
    output = []
    first_frame=True
    for frame in tqdm(frames):
        # run the bandwidth extender on the current frame
        print(frame.shape)
        # y = model(frame, sample_rate)
        # resampler=T.Resample(sample_rate,48000)
        # y=resampler(frame)
        y=frame
        print(y.shape)
        # fade out the previous frame, fade in the current
        if first_frame:
            y[:fade_samples] = y[:fade_samples] * fade_in
            first_frame=False
        else:
            y[:fade_samples] = prev * fade_ou + y[:fade_samples] * fade_in
        print(y.shape)
        # save off the previous frame for fading into the next
        # and add the current frame to the output
        prev = y[-fade_samples:]
        output.append(y[:-fade_samples])
        print("output length",len(torch.cat(output)))

    # tack on the fade out of the last frame
    output.append(prev)
    print("output length",len(torch.cat(output)))
    print("output length minus pad",len(torch.cat(output)[:-(pad_size)]))
    print("output length minus pad minus original",len(torch.cat(output)[:-(pad_size)])-len(x))
    return torch.cat(output)[:-pad_size]


if __name__ == "__main__":
    main()
