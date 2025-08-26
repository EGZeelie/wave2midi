# wave2midi

`wave2midi` is a Python tool that converts WAV audio files to MIDI format. It leverages a state-of-the-art source separation model, **Demucs**, to first separate the audio into individual instrument tracks (vocals, drums, bass, and other). This separation allows for more accurate pitch detection and results in cleaner, more usable MIDI files for each instrument.

## Features

-   **High-Quality Source Separation:** Uses Demucs to separate input WAV files into four distinct stems: vocals, drums, bass, and other.
-   **Accurate Note Detection:** Converts each isolated stem to a corresponding MIDI file with accurate note detection using `librosa`.
-   **Preserves Musical Information:** Retains timing and note velocity information in the output MIDI files.
-   **Configurable:** Allows for configuration of MIDI conversion parameters via a JSON file.

## How It Works

1.  **Stem Separation:** The input WAV file is processed by Demucs, which isolates the vocals, drums, bass, and other instrument tracks.
2.  **Audio Analysis:** Each stem undergoes pitch detection (`librosa.pyin`) to identify musical notes.
3.  **MIDI Conversion:** The detected notes are converted to MIDI events with appropriate timing, velocity, and instrument information using the `mido` library.
4.  **File Output:** Each stem produces a separate MIDI file, allowing for individual editing and remixing. For an input file `song.wav`, the tool will produce `song_vocals.mid`, `song_drums.mid`, `song_bass.mid`, and `song_other.mid`.

## Dependencies

-   Python 3.8+
-   [Demucs](https://github.com/adefossez/demucs) (and its dependencies, including `torch` and `torchaudio`)
-   Librosa
-   Mido
-   NumPy

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/wave2midi.git
    cd wave2midi
    ```

2.  **Install the package:**
    The project is packaged with `setup.py`. You can install it using `pip`.

    **For users:**
    ```bash
    pip install .
    ```
    This will install the `wave2midi` command and its dependencies. Note that this may take some time as it needs to download the Demucs models and PyTorch.

    **For developers:**
    If you want to install the project in editable mode and get the test dependencies, use the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command Line Interface

The simplest way to use `wave2midi` is via the command line.

**Basic Usage:**
```bash
wave2midi /path/to/your/song.wav /path/to/output_directory/
```

**With a custom configuration file:**
```bash
wave2midi song.wav output/ --config my_config.json
```

**With a different output BPM:**
```bash
wave2midi song.wav output/ --bpm 140
```

### Python API

You can also use the `WaveToMIDIConverter` class directly in your Python scripts.

```python
from wave2midi import WaveToMIDIConverter

# Create a converter instance
converter = WaveToMIDIConverter()

# Convert a WAV file to MIDI stems
midi_files = converter.convert("input.wav", "output_dir/")

print(f"Created {len(midi_files)} MIDI files from stems.")
```

## Configuration

You can configure the note detection and MIDI conversion parameters by creating a JSON file and passing it with the `--config` argument.

Here is an example `config.json`:
```json
{
  "sample_rate": 22050,
  "pitch_detection_method": "pyin",
  "frame_length": 2048,
  "hop_length": 512,
  "fmin": 27.5,
  "fmax": 4186.01,
  "probability_threshold": 0.5,
  "min_note_duration": 0.1,
  "max_note_duration": 2.0,
  "velocity_scaling": 1.0,
  "output_bpm": 120
}
```

## Limitations

-   Complex polyphonic music may result in note detection inaccuracies.
-   The quality of the source separation depends on the Demucs model and can vary.
-   Percussive elements may not always translate perfectly to MIDI percussion tracks.
-   Audio quality and recording conditions significantly affect conversion accuracy.

## Acknowledgments

-   The [Demucs](https://github.com/adefossez/demucs) team for their excellent source separation model.
-   The [librosa](https://librosa.org/) development team for their audio analysis tools.
-   The [mido](https://mido.readthedocs.io/) development team for their MIDI library.
