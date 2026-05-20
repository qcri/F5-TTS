import json
import os
from collections import Counter

from tqdm import tqdm


def create_emotion_dataset(root, dataset_metadata_output_path, dataset_type, asr_model=None, text_language="en"):
    if asr_model is None:
        from faster_whisper import WhisperModel

        faster_whisper_path = "ckpts/resources/models/models--Systran--faster-whisper-large-v2-local"
        model_bin_path = os.path.join(faster_whisper_path, "model.bin")
        if not os.path.isfile(model_bin_path):
            # Try snapshots path
            import glob

            candidates = glob.glob(os.path.join(faster_whisper_path, "snapshots", "*", "model.bin"))
            if candidates:
                model_bin_dir = os.path.dirname(candidates[0])
            else:
                raise FileNotFoundError(f"Cannot find model.bin under {faster_whisper_path}")
        else:
            model_bin_dir = faster_whisper_path
        asr_model = WhisperModel(model_bin_dir, device="cuda", compute_type="float16")

    def execute_asr(audio_path):
        segments, _ = asr_model.transcribe(
            audio=audio_path, beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700), language=text_language,
        )
        return "".join(segment.text for segment in segments)

    if dataset_type == "ESD":
        speaker_ids = [sid for sid in os.listdir(root) if os.path.isdir(os.path.join(root, sid))]
        emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
        phrases_dict = {}
        for speaker_id in speaker_ids:
            for emotion in emotions:
                emotion_dir = os.path.join(root, speaker_id, emotion)
                if not os.path.isdir(emotion_dir):
                    continue
                phrase_idx = 0
                for audio_name in sorted(os.listdir(emotion_dir)):
                    audio_path = os.path.join(emotion_dir, audio_name)
                    if phrase_idx not in phrases_dict:
                        phrases_dict[phrase_idx] = [audio_path]
                    else:
                        phrases_dict[phrase_idx].append(audio_path)
                    phrase_idx += 1

        dataset = {"ESD": []}
        for phrase_idx in tqdm(phrases_dict.keys()):
            transcription_texts = []
            for audio_path in phrases_dict[phrase_idx]:
                transcription_text = execute_asr(audio_path)
                transcription_texts.append(transcription_text)
            transcription_counts = Counter(transcription_texts)
            most_probable_transcription, _ = transcription_counts.most_common(1)[0]

            for audio_path in phrases_dict[phrase_idx]:
                emotion = os.path.basename(os.path.dirname(audio_path))
                speaker_id = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
                data_example = {
                    "phrase_idx": phrase_idx,
                    "audio_path": audio_path,
                    "text": most_probable_transcription,
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                    "text_alignment": [],
                }
                dataset["ESD"].append(data_example)

        with open(dataset_metadata_output_path, "w") as json_file:
            json.dump(dataset, json_file, indent=4)


if __name__ == "__main__":
    create_emotion_dataset("dataset/ESD/train", "dataset/ESD/train/dataset_descriptor.json", dataset_type="ESD")
