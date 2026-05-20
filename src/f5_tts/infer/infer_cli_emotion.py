from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torchaudio

from f5_tts.infer.infer_emotion import TTSModel, compute_mel_from_wav
from f5_tts.infer.utils_infer import cfg_strength, nfe_step, sway_sampling_coef
from f5_tts.model import CFMConditioned, DiTConditioned
from f5_tts.model.utils import get_tokenizer

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"
tokenizer = "pinyin"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CLI emotion inference using F5-TTS-Emotional-CFG.")

    p.add_argument("-ref", "--ref-audio-path", type=str, required=True, help="Path to reference .wav.")
    p.add_argument("-rt", "--ref-text", type=str, required=True, help="Transcription of reference audio.")
    p.add_argument(
        "-re", "--ref-emotion", type=str, default="Neutral",
        choices=["Angry", "Surprise", "Neutral", "Sad", "Happy"],
        help="Emotion label for the reference audio.",
    )

    p.add_argument("-it", "--inference-text", type=str, required=True, help="Text to synthesize.")
    p.add_argument(
        "-ie", "--inference-emotion", type=str, required=True,
        choices=["Angry", "Surprise", "Neutral", "Sad", "Happy"],
        help="Target emotion for generated speech.",
    )

    p.add_argument("-o", "--output-path", type=str, default="data/output.wav", help="Output .wav path.")
    p.add_argument("--checkpoint-path-emotion", type=str, default="ckpts/model_emo.pt", help="Emotion model checkpoint.")
    p.add_argument("--vocab-dataset-name", type=str, default="EmiliaPetite_dataset_ZH_EN")
    p.add_argument("--tokenizer", type=str, default=tokenizer, choices=["pinyin", "char", "custom"])
    p.add_argument("--tokenizer-path", type=str, default=None)

    p.add_argument("--nfe", type=int, default=nfe_step)
    p.add_argument("--cfg-strength", type=float, default=cfg_strength)
    p.add_argument("--cfg-strength2", type=float, default=10.0, help="Emotion guidance strength.")
    p.add_argument("--sway-sampling-coef", type=float, default=sway_sampling_coef)

    p.add_argument("--emotion-condition-type", type=str, default="text_mirror",
                    choices=["text_mirror", "cross_attention", "text_early_fusion", "film"])
    p.add_argument("--emotion-dim", type=int, default=128)
    p.add_argument("--emotion-conv-layers", type=int, default=4)
    p.add_argument("--init-type", type=str, default="xavier_reduced")
    p.add_argument("--weight-reduction-scale", type=float, default=1.0)

    p.add_argument("--mel-spec-type", type=str, default=mel_spec_type, choices=["vocos", "bigvgan"])
    p.add_argument("--target-sr", type=int, default=target_sample_rate)
    p.add_argument("--n-mel", type=int, default=n_mel_channels)
    p.add_argument("--n-fft", type=int, default=n_fft)
    p.add_argument("--hop-length", type=int, default=hop_length)
    p.add_argument("--win-length", type=int, default=win_length)

    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])

    return p


def main():
    args = build_arg_parser().parse_args()

    mel_spec_kwargs = dict(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.n_mel,
        target_sample_rate=args.target_sr,
        mel_spec_type=args.mel_spec_type,
    )

    if args.tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("tokenizer='custom' requires --tokenizer-path (vocab.txt).")
        vocab_char_map, vocab_size = get_tokenizer(args.vocab_dataset_name, args.tokenizer, args.tokenizer_path)
    else:
        vocab_char_map, vocab_size = get_tokenizer(args.vocab_dataset_name, args.tokenizer)

    emotion_conditioning_parameters = {
        "emotion_condition_type": args.emotion_condition_type,
        "init_type": args.init_type,
        "weight_reduction_scale": args.weight_reduction_scale,
        "emotion_dim": args.emotion_dim,
        "emotion_conv_layers": args.emotion_conv_layers,
        "load_emotion_weights": False,
    }

    model_cfg_emotion = dict(
        dim=1024, depth=22, heads=16, ff_mult=2,
        text_dim=512, emotion_dim=args.emotion_dim,
        conv_layers=args.emotion_conv_layers,
    )

    transformer = DiTConditioned(
        **model_cfg_emotion,
        text_num_embeds=vocab_size,
        mel_dim=args.n_mel,
        emotion_conditioning=emotion_conditioning_parameters,
    )

    model_emotion = CFMConditioned(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    tts = TTSModel(
        model=model_emotion,
        vocoder_name=args.mel_spec_type,
        checkpoint_path=args.checkpoint_path_emotion,
        emotion_conditioning_parameters=emotion_conditioning_parameters,
        device=args.device,
    )

    mel = compute_mel_from_wav(args.ref_audio_path, mel_spec_kwargs, device=args.device)

    start = time.perf_counter()
    gen_mel, gen_audio = tts.infer(
        inference_text=args.inference_text,
        inference_emotion=args.inference_emotion,
        ref_mel=mel,
        ref_text=args.ref_text,
        ref_emotion=args.ref_emotion,
        steps=args.nfe,
        cfg_strength=args.cfg_strength,
        cfg_strength2=args.cfg_strength2,
        sway_sampling_coef=args.sway_sampling_coef,
    )
    dur = time.perf_counter() - start

    outpath = Path(args.output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(outpath), gen_audio.cpu(), args.target_sr)

    print(f"[OK] Saved: {outpath}  |  duration: {dur:.2f}s")
    print(f"    Inference emotion: {args.inference_emotion}")
    print(f"    Steps (nfe): {args.nfe} | cfg_strength: {args.cfg_strength} | cfg_strength2: {args.cfg_strength2}")


if __name__ == "__main__":
    main()
