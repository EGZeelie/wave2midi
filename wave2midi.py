"""
wave2midi - Convert WAV files to MIDI by separating into stems
"""

import os
import sys
import argparse
import json
import warnings
from typing import List, Dict, Tuple, Optional
import numpy as np
import librosa
import mido
from mido import MidiFile, MidiTrack, Message
import tensorflow as tf
from spleeter.separator import Separator
from spleeter.utils.audio.adapter import AudioAdapter
from pathlib import Path

class WaveToMIDIConverter:
    """
    Convert WAV files to MIDI by first separating into stems and then
    converting each stem to MIDI using pitch detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the converter with optional configuration file.
        
        Args:
            config_path: Path to JSON configuration file
        """
        # Default configuration
        self.config = {
            "stem_count": 5,
            "sample_rate": 22050,
            "model_type": "spleeter:5stems",
            "pitch_detection_method": "pyin",
            "frame_length": 2048,
            "hop_length": 512,
            "fmin": 27.5,  # A0
            "fmax": 4186.01,  # C8
            "probability_threshold": 0.5,
            "min_note_duration": 0.1,
            "max_note_duration": 2.0,
            "velocity_scaling": 1.0,
            "output_bpm": 120,
            "instrument_mapping": {
                "vocals": 5,    # Voice
                "drums": 0,     # Acoustic Grand Piano (for percussion)
                "bass": 33,     # Electric Bass (finger)
                "piano": 1,     # Bright Acoustic Piano
                "guitar": 25,   # Acoustic Guitar (steel)
                "other": 40     # String Ensemble 1
            }
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self.config.update(user_config)
        
        # Set stem model based on stem count
        if self.config["stem_count"] == 2:
            self.config["model_type"] = "spleeter:2stems"
        elif self.config["stem_count"] == 4:
            self.config["model_type"] = "spleeter:4stems"
        elif self.config["stem_count"] == 5:
            self.config["model_type"] = "spleeter:5stems"
        
        # Initialize audio adapter
        self.audio_adapter = AudioAdapter.default()
        
        # Initialize separator
        try:
            self.separator = Separator(self.config["model_type"])
        except Exception as e:
            print(f"Warning: Could not initialize Spleeter model: {e}")
            print("Please make sure spleeter is installed: pip install spleeter")
            self.separator = None
    
    def separate_stems(self, wav_path: str) -> Dict[str, np.ndarray]:
        """
        Separate the input WAV file into stems.
        
        Args:
            wav_path: Path to input WAV file
            
        Returns:
            Dictionary with stem names as keys and audio data as values
        """
        if self.separator is None:
            raise RuntimeError("Separator not initialized. Please check Spleeter installation.")
        
        # Load audio
        print(f"Loading audio file: {wav_path}")
        try:
            waveform, sample_rate = self.audio_adapter.load(
                wav_path, 
                sample_rate=self.config["sample_rate"]
            )
        except Exception as e:
            raise RuntimeError(f"Could not load audio file {wav_path}: {e}")
        
        # Ensure mono or stereo (convert if needed)
        if len(waveform.shape) > 1:
            # Convert to mono by averaging channels if stereo
            if waveform.shape[0] > 1:
                waveform = np.mean(waveform, axis=0)
            else:
                waveform = waveform[0]
        
        # Perform separation
        print(f"Separating into {self.config['stem_count']} stems...")
        try:
            prediction = self.separator.separate(waveform)
        except Exception as e:
            raise RuntimeError(f"Stem separation failed: {e}")
        
        # Map stems based on model type
        stem_names = []
        if self.config["model_type"] == "spleeter:2stems":
            stem_names = ["vocals", "accompaniment"]
        elif self.config["model_type"] == "spleeter:4stems":
            stem_names = ["vocals", "drums", "bass", "other"]
        elif self.config["model_type"] == "spleeter:5stems":
            stem_names = ["vocals", "drums", "bass", "piano", "other"]
        
        # Extract audio data for each stem
        stems = {}
        for i, name in enumerate(stem_names):
            # Get the audio data (ensure mono)
            if len(prediction[name].shape) > 1:
                stem_audio = np.mean(prediction[name], axis=1)
            else:
                stem_audio = prediction[name]
            stems[name] = stem_audio
        
        return stems
    
    def detect_notes(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        """
        Detect notes in audio signal using pitch detection.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            List of detected notes with pitch, start_time, duration, and velocity
        """
        # Use pYIN for pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=self.config["fmin"],
            fmax=self.config["fmax"],
            sr=sample_rate,
            frame_length=self.config["frame_length"],
            hop_length=self.config["hop_length"]
        )
        
        # Convert frequency to MIDI notes
        midi_notes = []
        for i, (f, voiced, prob) in enumerate(zip(f0, voiced_flag, voiced_probs)):
            if voiced and prob > self.config["probability_threshold"] and f is not None:
                midi_pitch = int(round(librosa.hz_to_midi(f)))
                time = i * self.config["hop_length"] / sample_rate
                velocity = int(min(127, max(1, prob * 127 * self.config["velocity_scaling"])))
                
                midi_notes.append({
                    'pitch': midi_pitch,
                    'time': time,
                    'velocity': velocity,
                    'probability': prob
                })
        
        # Group notes into sustained notes
        final_notes = []
        active_notes = {}
        
        for note in midi_notes:
            pitch = note['pitch']
            time = note['time']
            velocity = note['velocity']
            
            if pitch not in active_notes:
                # Start new note
                active_notes[pitch] = {
                    'start_time': time,
                    'velocity': velocity
                }
            else:
                # Check if current note is significantly different in velocity
                # or if too much time has passed (gap in playing)
                time_diff = time - active_notes[pitch]['start_time']
                if time_diff > self.config["max_note_duration"]:
                    # End the previous note and start a new one
                    duration = min(time_diff, self.config["max_note_duration"])
                    if duration >= self.config["min_note_duration"]:
                        final_notes.append({
                            'pitch': pitch,
                            'start_time': active_notes[pitch]['start_time'],
                            'duration': duration,
                            'velocity': active_notes[pitch]['velocity']
                        })
                    # Start new note
                    active_notes[pitch] = {
                        'start_time': time,
                        'velocity': velocity
                    }
        
        # End any remaining active notes
        current_time = len(audio) / sample_rate
        for pitch, note_info in active_notes.items():
            duration = current_time - note_info['start_time']
            if duration >= self.config["min_note_duration"]:
                final_notes.append({
                    'pitch': pitch,
                    'start_time': note_info['start_time'],
                    'duration': min(duration, self.config["max_note_duration"]),
                    'velocity': note_info['velocity']
                })
        
        return final_notes
    
    def notes_to_midi(self, notes: List[Dict], instrument: str, tempo: int = 120) -> MidiFile:
        """
        Convert detected notes to MIDI file.
        
        Args:
            notes: List of detected notes
            instrument: Instrument name for program change
            tempo: Tempo in BPM
            
        Returns:
            MIDI file object
        """
        midi = MidiFile()
        track = MidiTrack()
        midi.tracks.append(track)
        
        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Program change based on instrument
        program = self.config["instrument_mapping"].get(instrument, 0)
        track.append(Message('program_change', channel=0, program=program))
        
        # Sort notes by start time
        notes.sort(key=lambda x: x['start_time'])
        
        # Current time in ticks
        current_tick = 0
        
        # Create note on/off messages
        for note in notes:
            # Calculate ticks for start time
            start_tick = int(mido.time_to_ticks(
                note['start_time'], 
                midi.ticks_per_beat, 
                midi.meta_track[0].tempo
            ))
            
            # Calculate ticks for duration
            duration_tick = int(mido.time_to_ticks(
                note['duration'], 
                midi.ticks_per_beat, 
                midi.meta_track[0].tempo
            ))
            
            # Add wait time if needed
            wait_tick = start_tick - current_tick
            if wait_tick > 0:
                track.append(Message('note_on', note=0, velocity=0, time=wait_tick))
            
            # Note on
            track.append(Message('note_on', note=note['pitch'], 
                               velocity=note['velocity'], time=0))
            
            # Note off
            track.append(Message('note_off', note=note['pitch'], 
                               velocity=0, time=duration_tick))
            
            current_tick = start_tick + duration_tick
        
        return midi
    
    def convert(self, wav_path: str, output_dir: str) -> List[str]:
        """
        Convert a WAV file to multiple MIDI files by separating into stems.
        
        Args:
            wav_path: Path to input WAV file
            output_dir: Directory to save output MIDI files
            
        Returns:
            List of paths to created MIDI files
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Input file not found: {wav_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        
        # Separate into stems
        stems = self.separate_stems(wav_path)
        
        # Convert each stem to MIDI
        midi_files = []
        for stem_name, stem_audio in stems.items():
            print(f"Processing {stem_name} stem...")
            
            # Detect notes in the stem
            notes = self.detect_notes(stem_audio, self.config["sample_rate"])
            
            if not notes:
                print(f"No notes detected in {stem_name} stem")
                continue
                
            # Convert notes to MIDI
            midi = self.notes_to_midi(notes, stem_name, self.config["output_bpm"])
            
            # Save MIDI file
            midi_filename = f"{base_name}_{stem_name}.mid"
            midi_path = os.path.join(output_dir, midi_filename)
            midi.save(midi_path)
            
            print(f"Saved {midi_path} ({len(notes)} notes)")
            midi_files.append(midi_path)
        
        return midi_files

def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert WAV files to MIDI by separating into stems"
    )
    parser.add_argument("input_wav", help="Input WAV file path")
    parser.add_argument("output_dir", help="Output directory for MIDI files")
    parser.add_argument(
        "--stems", 
        type=int, 
        choices=[2, 4, 5], 
        default=5,
        help="Number of stems to separate (2, 4, or 5)"
    )
    parser.add_argument(
        "--config", 
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--bpm", 
        type=int, 
        default=120,
        help="Output tempo in BPM"
    )
    
    args = parser.parse_args()
    
    try:
        # Create converter
        converter = WaveToMIDIConverter(args.config)
        
        # Update stem count if specified
        if args.stems:
            converter.config["stem_count"] = args.stems
            if args.stems == 2:
                converter.config["model_type"] = "spleeter:2stems"
            elif args.stems == 4:
                converter.config["model_type"] = "spleeter:4stems"
            elif args.stems == 5:
                converter.config["model_type"] = "spleeter:5stems"
        
        # Update BPM if specified
        if args.bpm:
            converter.config["output_bpm"] = args.bpm
        
        # Perform conversion
        print(f"Converting {args.input_wav} to MIDI...")
        midi_files = converter.convert(args.input_wav, args.output_dir)
        
        print(f"\nConversion complete!")
        print(f"Created {len(midi_files)} MIDI files:")
        for midi_file in midi_files:
            print(f"  {midi_file}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()