from __future__ import annotations

import gc
import time

import torch
import torchaudio
from ema_pytorch import EMA

from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef
from f5_tts.model import CFM, CFMConditioned, DiT, DiTConditioned, UNetT
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import get_tokenizer

# ----------------------- Dataset Settings ----------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"

tokenizer = "pinyin"
tokenizer_path = None

emotion_dict = {
    "Angry": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
    "Happy": 5,
}

wandb_resume_id = None
model_cls_emotion = DiTConditioned
model_cls_pretrained = DiT
model_cfg_pretrained = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


def compute_mel_from_wav(audio_path: str, mel_spec_kwargs: dict, device: str = "cpu") -> torch.Tensor:
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sample_rate != mel_spec_kwargs["target_sample_rate"]:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=mel_spec_kwargs["target_sample_rate"]
        )
        audio = resampler(audio)

    audio = audio.to(device)
    mel_processor = MelSpec(**mel_spec_kwargs).to(device)
    mel = mel_processor(audio)
    return mel.squeeze(0).permute(1, 0)


class TTSModel:
    def __init__(self, model, vocoder_name, checkpoint_path: str, emotion_conditioning_parameters, device: str = "cuda"):
        self.device = device
        self.model = model
        self._load_checkpoint(checkpoint_path)
        self.emotion_conditioning_parameters = emotion_conditioning_parameters
        self.vocoder_name = vocoder_name
        self.vocoder = load_vocoder(vocoder_name=self.vocoder_name)

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")

        if "step" in checkpoint:
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        del checkpoint
        gc.collect()

    def remove_leading_value(self, spec, value=0.0):
        gen_flat = spec[0]
        is_row_of_ones = torch.all(gen_flat == value, dim=1)
        num_rows_to_remove = torch.sum(is_row_of_ones).item()
        spec = spec[:, num_rows_to_remove:, :]
        return spec

    @torch.inference_mode()
    def infer(
        self,
        inference_text: str,
        inference_emotion: str,
        ref_mel: torch.Tensor,
        ref_text: str,
        ref_emotion: str,
        steps: int,
        cfg_strength,
        cfg_strength2,
        sway_sampling_coef,
        seed: int = 50,
    ) -> torch.Tensor:
        text_input = [ref_text + " " + inference_text]
        emotion_input = [[ref_emotion, inference_emotion]]
        first_phrase_length = [len(ref_text)]

        mel_lengths = torch.LongTensor([ref_mel.shape[0]])
        ref_audio_len = mel_lengths.item()
        estimated_duration = ref_audio_len + int(ref_audio_len * len(inference_text) / len(ref_text))

        start = time.perf_counter()
        if inference_emotion is not None:
            generated_melspec, _ = self.model.sample(
                cond=ref_mel.to(self.device).unsqueeze(0),
                text=text_input,
                emotion=emotion_input,
                first_phrase_length=first_phrase_length,
                duration=estimated_duration,
                steps=steps,
                cfg_strength=cfg_strength,
                cfg_strength2=cfg_strength2,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        else:
            generated_melspec, _ = self.model.sample(
                cond=ref_mel.to(self.device).unsqueeze(0),
                text=text_input,
                duration=estimated_duration,
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        end = time.perf_counter()
        print(f"TIME sampling ({len(text_input[0])}): {end - start:.2f}s")

        generated_melspec = self.remove_leading_value(generated_melspec)
        generated_melspec_2ndhalf = generated_melspec[:, ref_mel.shape[0] :, :]

        start = time.perf_counter()
        generated_audio = self.vocode(generated_melspec_2ndhalf)
        end = time.perf_counter()
        print(f"TIME vocoder ({len(text_input[0])}): {end - start:.2f}s")

        return generated_melspec_2ndhalf, generated_audio

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.unsqueeze(0) if mel.ndim == 2 else mel
        return self.vocoder.decode(mel.float().permute(0, 2, 1).to(self.device))


if __name__ == "__main__":
    ref_audio_path = "data/0011_angry.wav"
    ref_emotion = "Angry"
    ref_text = "The nine, the eggs, I keep."

    inference_text = "Hello, this is a text to check emotion."
    inference_emotion = "Surprise"
    output_path = "data/output.wav"

    nfe = nfe_step
    cfg_strength2 = 10

    emotion_conditioning_parameters = {
        "emotion_condition_type": "text_mirror",
        "init_type": "xavier_reduced",
        "weight_reduction_scale": 1,
        "emotion_dim": 128,
        "emotion_conv_layers": 4,
        "load_emotion_weights": False,
    }

    vocab_char_map, vocab_size = get_tokenizer("EmiliaPetite_dataset_ZH_EN", "pinyin")
    device = "cuda"

    checkpoint_path_emotion = "ckpts/model_emo.pt"
    checkpoint_path_pretrained = "ckpts/model_0.pt"

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    emotion_dim = emotion_conditioning_parameters["emotion_dim"]
    emotion_conv_layers = emotion_conditioning_parameters["emotion_conv_layers"]
    model_cfg_emotion = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
        emotion_dim=emotion_dim, conv_layers=emotion_conv_layers,
    )

    model_emotion = CFMConditioned(
        transformer=model_cls_emotion(
            **model_cfg_emotion,
            text_num_embeds=vocab_size,
            mel_dim=n_mel_channels,
            emotion_conditioning=emotion_conditioning_parameters,
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    model_wrapper_emotion = TTSModel(
        model_emotion, mel_spec_type, checkpoint_path_emotion, emotion_conditioning_parameters, device
    )

    model_pretrained = CFM(
        transformer=model_cls_pretrained(
            **model_cfg_pretrained, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    model_wrapper_pretrained = TTSModel(
        model_pretrained, mel_spec_type, checkpoint_path_pretrained, emotion_conditioning_parameters, device
    )

    mel = compute_mel_from_wav(ref_audio_path, mel_spec_kwargs, device="cuda")

    generated_melspec, generated_audio = model_wrapper_emotion.infer(
        inference_text=inference_text,
        inference_emotion=inference_emotion,
        ref_mel=mel,
        ref_text=ref_text,
        ref_emotion=ref_emotion,
        steps=nfe,
        cfg_strength=cfg_strength,
        cfg_strength2=cfg_strength2,
        sway_sampling_coef=sway_sampling_coef,
    )
    torchaudio.save(output_path.replace(".wav", f"_{inference_emotion}.wav"), generated_audio.cpu(), target_sample_rate)

    generated_melspec, generated_audio = model_wrapper_pretrained.infer(
        inference_text=inference_text,
        inference_emotion=None,
        ref_mel=mel,
        ref_text=ref_text,
        ref_emotion=None,
        steps=nfe,
        cfg_strength=cfg_strength,
        cfg_strength2=None,
        sway_sampling_coef=sway_sampling_coef,
    )
    torchaudio.save(output_path.replace(".wav", "_NOemotion.wav"), generated_audio.cpu(), target_sample_rate)
