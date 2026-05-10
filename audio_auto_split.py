import torch
import torchaudio
import torch.nn.functional as F
from typing import List, Tuple, Optional, Callable
import math


class AudioAutoSplitProcessor:
    """
    音频自动分割处理器
    
    功能：
    1. 按指定最大时长自动分割长音频
    2. 对每个片段进行循环处理
    3. 使用交叉淡入淡出智能合并结果
    
    适用场景：
    - TTS 参考音频过长时自动分割处理
    - 音频分离、降噪等需要限制输入长度的场景
    - 批量处理长音频并合并结果
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_segment_duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 300.0,
                    "step": 1.0,
                    "display": "number",
                    "tooltip": "每段音频的最大时长（秒），超过此值将自动分割"
                }),
                "overlap_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "相邻片段之间的重叠时长（秒），用于交叉淡入淡出合并"
                }),
                "fade_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "淡入淡出的时长（秒）"
                }),
                "min_segment_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                    "display": "number",
                    "tooltip": "最小片段时长（秒），最后一段如果小于此值将合并到前一段"
                }),
            },
            "optional": {
                "enable_split": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用自动分割功能"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "segment_count", "segment_info")
    FUNCTION = "process_audio"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def __init__(self):
        self.segment_info = []

    def split_audio(self, waveform: torch.Tensor, sample_rate: int, 
                    max_segment_duration: float, overlap_duration: float,
                    min_segment_duration: float) -> List[Tuple[torch.Tensor, float, float]]:
        """
        将音频分割成多个片段
        
        Args:
            waveform: 音频波形 [channels, samples]
            sample_rate: 采样率
            max_segment_duration: 最大片段时长（秒）
            overlap_duration: 重叠时长（秒）
            min_segment_duration: 最小片段时长（秒）
            
        Returns:
            List of (segment_waveform, start_time, end_time)
        """
        total_samples = waveform.shape[-1]
        total_duration = total_samples / sample_rate
        
        # 如果音频时长小于最大片段时长，不分割
        if total_duration <= max_segment_duration:
            return [(waveform, 0.0, total_duration)]
        
        max_segment_samples = int(max_segment_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        min_segment_samples = int(min_segment_duration * sample_rate)
        
        segments = []
        start_sample = 0
        segment_idx = 0
        
        while start_sample < total_samples:
            # 计算当前片段的结束位置
            end_sample = min(start_sample + max_segment_samples, total_samples)
            
            # 如果不是最后一段，检查剩余部分是否大于最小片段时长
            if end_sample < total_samples:
                remaining_samples = total_samples - end_sample
                if remaining_samples < min_segment_samples:
                    # 剩余部分太小，合并到当前段
                    end_sample = total_samples
            
            # 提取片段
            segment = waveform[..., start_sample:end_sample]
            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate
            
            segments.append((segment, start_time, end_time))
            
            # 下一段的起始位置（考虑重叠）
            if end_sample >= total_samples:
                break
                
            start_sample = end_sample - overlap_samples
            segment_idx += 1
            
            # 安全检查，防止无限循环
            if segment_idx > 1000:
                print("警告：音频分割片段过多，强制停止")
                break
        
        return segments

    def apply_fade(self, waveform: torch.Tensor, sample_rate: int, 
                   fade_in_duration: float, fade_out_duration: float) -> torch.Tensor:
        """
        对音频应用淡入淡出
        
        Args:
            waveform: 音频波形 [channels, samples]
            sample_rate: 采样率
            fade_in_duration: 淡入时长（秒）
            fade_out_duration: 淡出时长（秒）
            
        Returns:
            处理后的音频
        """
        num_samples = waveform.shape[-1]
        fade_in_samples = int(fade_in_duration * sample_rate)
        fade_out_samples = int(fade_out_duration * sample_rate)
        
        # 创建增益曲线
        gain = torch.ones(num_samples, device=waveform.device, dtype=waveform.dtype)
        
        # 淡入
        if fade_in_samples > 0 and fade_in_samples < num_samples:
            fade_in_curve = torch.linspace(0, 1, fade_in_samples, device=waveform.device, dtype=waveform.dtype)
            gain[:fade_in_samples] = fade_in_curve
        
        # 淡出
        if fade_out_samples > 0 and fade_out_samples < num_samples:
            fade_out_curve = torch.linspace(1, 0, fade_out_samples, device=waveform.device, dtype=waveform.dtype)
            gain[-fade_out_samples:] = fade_out_curve
        
        # 应用增益
        return waveform * gain.unsqueeze(0)

    def crossfade_merge(self, segments: List[torch.Tensor], sample_rate: int, 
                        overlap_duration: float) -> torch.Tensor:
        """
        使用交叉淡入淡出合并多个音频片段
        
        Args:
            segments: 音频片段列表，每个片段 [channels, samples]
            sample_rate: 采样率
            overlap_duration: 重叠时长（秒）
            
        Returns:
            合并后的音频
        """
        if len(segments) == 0:
            return torch.zeros(1, 0)
        
        if len(segments) == 1:
            return segments[0]
        
        overlap_samples = int(overlap_duration * sample_rate)
        
        # 确定输出通道数
        max_channels = max(seg.shape[0] for seg in segments)
        
        # 统一通道数
        normalized_segments = []
        for seg in segments:
            if seg.shape[0] < max_channels:
                # 单声道转多声道
                if seg.shape[0] == 1:
                    seg = seg.repeat(max_channels, 1)
                else:
                    # 填充通道
                    padding = max_channels - seg.shape[0]
                    seg = torch.cat([seg, seg[-1:].repeat(padding, 1)], dim=0)
            normalized_segments.append(seg)
        
        # 计算总长度
        total_length = sum(seg.shape[-1] for seg in normalized_segments)
        total_length -= overlap_samples * (len(normalized_segments) - 1)
        
        # 创建输出缓冲区
        output = torch.zeros(max_channels, total_length, 
                           device=normalized_segments[0].device, 
                           dtype=normalized_segments[0].dtype)
        
        # 当前写入位置
        current_pos = 0
        
        for i, segment in enumerate(normalized_segments):
            seg_length = segment.shape[-1]
            
            if i == 0:
                # 第一段：直接写入
                output[:, :seg_length] = segment
                current_pos = seg_length
            else:
                # 后续段：与前面重叠部分进行交叉淡入淡出
                if overlap_samples > 0 and current_pos > overlap_samples:
                    # 重叠区域
                    overlap_start = current_pos - overlap_samples
                    overlap_end = min(current_pos, overlap_start + seg_length)
                    actual_overlap = overlap_end - overlap_start

                    if actual_overlap > 0:
                        # 创建淡入淡出曲线
                        fade_out = torch.linspace(1, 0, actual_overlap, 
                                                device=output.device, 
                                                dtype=output.dtype)
                        fade_in = torch.linspace(0, 1, actual_overlap,
                                               device=output.device,
                                               dtype=output.dtype)

                        # 重叠区域混合
                        existing = output[:, overlap_start:overlap_end]
                        incoming = segment[:, :actual_overlap]

                        output[:, overlap_start:overlap_end] = (
                            existing * fade_out.unsqueeze(0) +
                            incoming * fade_in.unsqueeze(0)
                        )

                        # 写入非重叠部分
                        if seg_length > actual_overlap:
                            output[:, overlap_end:overlap_end + seg_length - actual_overlap] = (
                                segment[:, actual_overlap:]
                            )

                        current_pos = overlap_end + max(0, seg_length - actual_overlap)
                    else:
                        # 无实际重叠，直接追加
                        end_pos = current_pos + seg_length
                        output[:, current_pos:end_pos] = segment
                        current_pos = end_pos
                else:
                    # 无重叠，直接追加
                    end_pos = current_pos + seg_length
                    output[:, current_pos:end_pos] = segment
                    current_pos = end_pos
        
        # 裁剪到实际长度
        output = output[:, :current_pos]
        
        return output

    def process_audio(self, audio, max_segment_duration=30.0, overlap_duration=1.0,
                     fade_duration=0.5, min_segment_duration=5.0, enable_split=True):
        """
        主处理函数：分割音频并返回片段信息
        
        注意：此节点只负责分割，实际处理需要在下游节点中对每个片段进行处理
        然后使用 AudioMergeSegments 节点合并结果
        
        Args:
            audio: 输入音频 {"waveform": tensor, "sample_rate": int}
            max_segment_duration: 最大片段时长
            overlap_duration: 重叠时长
            fade_duration: 淡入淡出时长
            min_segment_duration: 最小片段时长
            enable_split: 是否启用分割
            
        Returns:
            audio: 第一个片段（用于兼容连接）
            segment_count: 片段数量
            segment_info: 片段信息 JSON 字符串
        """
        waveform = audio["waveform"].squeeze(0)  # [channels, samples]
        sample_rate = audio["sample_rate"]
        
        total_duration = waveform.shape[-1] / sample_rate
        
        if not enable_split or total_duration <= max_segment_duration:
            # 不需要分割
            info = {
                "total_duration": total_duration,
                "segment_count": 1,
                "segments": [
                    {
                        "index": 0,
                        "start_time": 0.0,
                        "end_time": total_duration,
                        "duration": total_duration,
                        "samples": waveform.shape[-1]
                    }
                ]
            }
            return (audio, 1, str(info))
        
        # 分割音频
        segments = self.split_audio(
            waveform, sample_rate, max_segment_duration, 
            overlap_duration, min_segment_duration
        )
        
        # 构建片段信息
        segment_list = []
        for i, (seg_waveform, start_time, end_time) in enumerate(segments):
            segment_info = {
                "index": i,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "samples": seg_waveform.shape[-1]
            }
            segment_list.append(segment_info)
        
        info = {
            "total_duration": total_duration,
            "segment_count": len(segments),
            "overlap_duration": overlap_duration,
            "segments": segment_list
        }
        
        # 返回第一个片段作为音频输出（用于兼容）
        first_segment = segments[0][0].unsqueeze(0)  # [1, channels, samples]
        output_audio = {
            "waveform": first_segment,
            "sample_rate": sample_rate
        }
        
        return (output_audio, len(segments), str(info))


class AudioSegmentExtractor:
    """
    音频片段提取器
    
    根据索引从分割信息中提取指定片段
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "segment_info": ("STRING", {"forceInput": True}),
                "segment_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999,
                    "step": 1,
                    "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "start_time", "end_time")
    FUNCTION = "extract_segment"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def extract_segment(self, audio, segment_info, segment_index):
        """
        提取指定索引的音频片段
        
        Args:
            audio: 原始音频
            segment_info: 片段信息字符串（JSON格式）
            segment_index: 片段索引
            
        Returns:
            audio: 提取的片段
            start_time: 开始时间
            end_time: 结束时间
        """
        import ast
        
        waveform = audio["waveform"].squeeze(0)  # [channels, samples]
        sample_rate = audio["sample_rate"]
        
        # 解析片段信息
        try:
            info = ast.literal_eval(segment_info)
        except:
            # 如果解析失败，返回整个音频
            return (audio, 0.0, waveform.shape[-1] / sample_rate)
        
        segments = info.get("segments", [])
        
        if segment_index >= len(segments):
            print(f"警告：片段索引 {segment_index} 超出范围，返回整个音频")
            return (audio, 0.0, waveform.shape[-1] / sample_rate)
        
        seg_info = segments[segment_index]
        start_time = seg_info["start_time"]
        end_time = seg_info["end_time"]
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # 提取片段
        segment_waveform = waveform[..., start_sample:end_sample]
        
        # 应用淡入淡出
        fade_samples = int(0.1 * sample_rate)  # 0.1秒淡入淡出
        if segment_waveform.shape[-1] > fade_samples * 2:
            fade_in = torch.linspace(0, 1, fade_samples, device=segment_waveform.device, dtype=segment_waveform.dtype)
            fade_out = torch.linspace(1, 0, fade_samples, device=segment_waveform.device, dtype=segment_waveform.dtype)
            segment_waveform[..., :fade_samples] *= fade_in.unsqueeze(0)
            segment_waveform[..., -fade_samples:] *= fade_out.unsqueeze(0)
        
        output_audio = {
            "waveform": segment_waveform.unsqueeze(0),
            "sample_rate": sample_rate
        }
        
        return (output_audio, start_time, end_time)


class AudioMergeSegments:
    """
    音频片段合并器
    
    使用交叉淡入淡出合并多个处理后的音频片段
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segment_count": ("INT", {"forceInput": True}),
                "overlap_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number",
                    "tooltip": "片段之间的重叠时长（秒），应与分割时的值一致"
                }),
            },
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
                "audio_3": ("AUDIO",),
                "audio_4": ("AUDIO",),
                "audio_5": ("AUDIO",),
                "audio_6": ("AUDIO",),
                "audio_7": ("AUDIO",),
                "audio_8": ("AUDIO",),
                "audio_9": ("AUDIO",),
                "audio_10": ("AUDIO",),
                "audio_11": ("AUDIO",),
                "audio_12": ("AUDIO",),
                "audio_13": ("AUDIO",),
                "audio_14": ("AUDIO",),
                "audio_15": ("AUDIO",),
                "audio_16": ("AUDIO",),
                "audio_17": ("AUDIO",),
                "audio_18": ("AUDIO",),
                "audio_19": ("AUDIO",),
                "audio_20": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "merge_segments"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def crossfade_merge(self, segments: List[torch.Tensor], sample_rate: int, 
                        overlap_duration: float) -> torch.Tensor:
        """
        使用交叉淡入淡出合并音频片段
        """
        if len(segments) == 0:
            return torch.zeros(1, 0)
        
        if len(segments) == 1:
            return segments[0]
        
        overlap_samples = int(overlap_duration * sample_rate)
        
        # 确定输出通道数
        max_channels = max(seg.shape[0] for seg in segments)
        
        # 统一通道数
        normalized_segments = []
        for seg in segments:
            if seg.shape[0] < max_channels:
                if seg.shape[0] == 1:
                    seg = seg.repeat(max_channels, 1)
                else:
                    padding = max_channels - seg.shape[0]
                    seg = torch.cat([seg, seg[-1:].repeat(padding, 1)], dim=0)
            normalized_segments.append(seg)
        
        # 计算总长度
        total_length = sum(seg.shape[-1] for seg in normalized_segments)
        total_length -= overlap_samples * (len(normalized_segments) - 1)
        
        # 创建输出缓冲区
        output = torch.zeros(max_channels, total_length,
                           device=normalized_segments[0].device,
                           dtype=normalized_segments[0].dtype)
        
        current_pos = 0
        
        for i, segment in enumerate(normalized_segments):
            seg_length = segment.shape[-1]
            
            if i == 0:
                output[:, :seg_length] = segment
                current_pos = seg_length
            else:
                if overlap_samples > 0 and current_pos > overlap_samples:
                    overlap_start = current_pos - overlap_samples
                    overlap_end = min(current_pos, overlap_start + seg_length)
                    actual_overlap = overlap_end - overlap_start
                    
                    if actual_overlap > 0:
                        fade_out = torch.linspace(1, 0, actual_overlap,
                                                device=output.device,
                                                dtype=output.dtype)
                        fade_in = torch.linspace(0, 1, actual_overlap,
                                               device=output.device,
                                               dtype=output.dtype)
                        
                        existing = output[:, overlap_start:overlap_end]
                        incoming = segment[:, :actual_overlap]
                        
                        output[:, overlap_start:overlap_end] = (
                            existing * fade_out.unsqueeze(0) +
                            incoming * fade_in.unsqueeze(0)
                        )
                        
                        if seg_length > actual_overlap:
                            output[:, overlap_end:overlap_end + seg_length - actual_overlap] = (
                                segment[:, actual_overlap:]
                            )
                        
                        current_pos = overlap_end + max(0, seg_length - actual_overlap)
                    else:
                        end_pos = current_pos + seg_length
                        output[:, current_pos:end_pos] = segment
                        current_pos = end_pos
                else:
                    end_pos = current_pos + seg_length
                    output[:, current_pos:end_pos] = segment
                    current_pos = end_pos
        
        output = output[:, :current_pos]
        return output

    def merge_segments(self, segment_count, overlap_duration=1.0,
                      audio_1=None, audio_2=None, audio_3=None, audio_4=None,
                      audio_5=None, audio_6=None, audio_7=None, audio_8=None,
                      audio_9=None, audio_10=None, audio_11=None, audio_12=None,
                      audio_13=None, audio_14=None, audio_15=None, audio_16=None,
                      audio_17=None, audio_18=None, audio_19=None, audio_20=None):
        """
        合并多个音频片段
        
        Args:
            segment_count: 片段数量
            overlap_duration: 重叠时长
            audio_1...audio_20: 音频片段输入
            
        Returns:
            合并后的音频
        """
        # 收集所有非空音频片段
        audio_inputs = [
            audio_1, audio_2, audio_3, audio_4, audio_5,
            audio_6, audio_7, audio_8, audio_9, audio_10,
            audio_11, audio_12, audio_13, audio_14, audio_15,
            audio_16, audio_17, audio_18, audio_19, audio_20
        ]
        
        segments = []
        sample_rate = None
        
        for i, audio in enumerate(audio_inputs[:segment_count]):
            if audio is not None:
                waveform = audio["waveform"].squeeze(0)  # [channels, samples]
                segments.append(waveform)
                if sample_rate is None:
                    sample_rate = audio["sample_rate"]
        
        if len(segments) == 0:
            # 返回空音频
            empty_audio = {
                "waveform": torch.zeros(1, 0).unsqueeze(0),
                "sample_rate": sample_rate if sample_rate else 44100
            }
            return (empty_audio,)
        
        # 合并片段
        merged_waveform = self.crossfade_merge(segments, sample_rate, overlap_duration)
        
        output_audio = {
            "waveform": merged_waveform.unsqueeze(0),
            "sample_rate": sample_rate
        }
        
        return (output_audio,)


class AudioBatchProcessor:
    """
    音频批处理器
    
    自动将长音频分割、批量处理、然后合并结果
    这是一个高级节点，内部自动完成分割-处理-合并的全流程
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_segment_duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 300.0,
                    "step": 1.0,
                    "tooltip": "每段最大时长（秒）"
                }),
                "overlap_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "重叠时长（秒）"
                }),
                "process_type": (["separation", "denoising", "custom"], {
                    "default": "separation",
                    "tooltip": "处理类型（预留，当前需要配合其他节点使用）"
                }),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_segments", "batch_info")
    FUNCTION = "batch_process"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def batch_process(self, audio, max_segment_duration=30.0, 
                     overlap_duration=1.0, process_type="separation"):
        """
        批处理音频，返回分割后的片段列表信息
        
        注意：实际处理需要在下游节点中完成
        """
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        total_duration = waveform.shape[-1] / sample_rate
        
        # 计算分割参数
        max_samples = int(max_segment_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        segments_info = {
            "original_duration": total_duration,
            "sample_rate": sample_rate,
            "max_segment_duration": max_segment_duration,
            "overlap_duration": overlap_duration,
            "segments": []
        }
        
        if total_duration <= max_segment_duration:
            segments_info["segments"].append({
                "index": 0,
                "start": 0,
                "end": waveform.shape[-1],
                "start_time": 0.0,
                "end_time": total_duration
            })
        else:
            start = 0
            idx = 0
            while start < waveform.shape[-1]:
                end = min(start + max_samples, waveform.shape[-1])
                
                segments_info["segments"].append({
                    "index": idx,
                    "start": start,
                    "end": end,
                    "start_time": start / sample_rate,
                    "end_time": end / sample_rate
                })
                
                if end >= waveform.shape[-1]:
                    break
                    
                start = end - overlap_samples
                idx += 1
                
                if idx > 1000:
                    break
        
        # 返回原始音频和分割信息
        return (audio, str(segments_info))



