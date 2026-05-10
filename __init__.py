from .audiotoolsnode import *
from .minimal_pause_node import MinimalPauseNode
from .AddSubtitlesToVideo import AddSubtitlesToTensor
from .MWAudioRecorderAT import AudioRecorderAT
from .clearvoicenode import ClearVoiceRun
from .audio_separate import MusicSeparation, SpeechSeparation, MergeAudioMW
from .audio_auto_split import (
    AudioAutoSplitProcessor,
    AudioSegmentExtractor,
    AudioMergeSegments,
    AudioBatchProcessor,
)


NODE_CLASS_MAPPINGS = {
    "ClearVoiceRun": ClearVoiceRun,
    "LoadAudioMW": LoadAudioMW,
    "MinimalPauseNode": MinimalPauseNode,
    "AudioConcatenate": AudioConcatenate,
    "AudioAddWatermark": AudioAddWatermark,
    "AdjustAudio": AdjustAudio,
    "TrimAudio": TrimAudio,
    "RemoveSilence": RemoveSilence,
    "AudioRecorderAT": AudioRecorderAT,
    "AddSubtitlesToVideo": AddSubtitlesToTensor,
    "MultiLinePromptAT": MultiLinePromptAT,
    "StringEditNode": StringEditNode,
    "MusicSeparation": MusicSeparation,
    "SpeechSeparation": SpeechSeparation,
    "MergeAudioMW": MergeAudioMW,
    "AudioAutoSplitProcessor": AudioAutoSplitProcessor,
    "AudioSegmentExtractor": AudioSegmentExtractor,
    "AudioMergeSegments": AudioMergeSegments,
    "AudioBatchProcessor": AudioBatchProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearVoiceRun": "Clear Voice @MW",
    "LoadAudioMW": "Load Audio @MW",
    "MinimalPauseNode": "Pause Node @MW",
    "AudioConcatenate": "Audio Concatenate",
    "AudioAddWatermark": "Audio Watermark Embedding",
    "AdjustAudio": "Adjust Audio",
    "TrimAudio": "Trim Audio",
    "RemoveSilence": "Remove Silence",
    "AudioRecorderAT": "MW Audio Recorder",
    "AddSubtitlesToVideo": "Add Subtitles To Video",
    "MultiLinePromptAT": "Multi-Line Prompt",
    "StringEditNode": "String Edit",
    "MusicSeparation": "Music Separation",
    "SpeechSeparation": "Speech Separation",
    "MergeAudioMW": "Merge Audio",
    "AudioAutoSplitProcessor": "Audio Auto Split Processor (音频自动分割)",
    "AudioSegmentExtractor": "Audio Segment Extractor (音频片段提取)",
    "AudioMergeSegments": "Audio Merge Segments (音频片段合并)",
    "AudioBatchProcessor": "Audio Batch Processor (音频批处理)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"
