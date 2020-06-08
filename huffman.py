from collections import defaultdict
import torch
from dahuffman import HuffmanCodec

def encode(data, fps=10):
    assert len(data.shape) == 4
    num_frames = data.shape[0]
    flattened = torch.flatten(data).sign()
    print(flattened.shape)
    flattened[flattened == 0] = 1
    data = flattened.int().tolist()

    freqs = {}
    freqs[1] = torch.sum(flattened == 1).item()
    freqs[-1] = torch.sum(flattened == -1).item()
    assert freqs[1] + freqs[-1] == len(data)

    codec = HuffmanCodec.from_frequencies(freqs)
    encoded = codec.encode(data)
    added_bitrate = (len(encoded) + (len(freqs) * (32 * 2 * 8))) / (num_frames * (2 ** 20)) * fps
    return freqs, encoded, added_bitrate

def decode(encoded, freqs, shape):
    codec = HuffmanCodec.from_frequencies(freqs)
    decoded = torch.Tensor(codec.decode(encoded))
    return torch.reshape(decoded, shape)
