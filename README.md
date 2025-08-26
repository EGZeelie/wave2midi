# Audio Processing Music Technology Python

wave2midi is a tool that converts WAV audio files to MIDI format by first separating the audio into 5-7 stems (individual instrument/vocal tracks) and then converting each stem to a separate MIDI file. This approach allows for more accurate note detection by isolating different instrument types before pitch and note analysis.

## Features
Separates input WAV files into 5-7 distinct stems (e.g., vocals, drums, bass, piano, guitar, etc.)
Converts each isolated stem to a corresponding MIDI file with accurate note detection
Preserves timing and note velocity information in the output MIDI files
Supports batch processing of multiple WAV files
Configurable stem separation and MIDI conversion parameters
Installation
To install wave2midi, follow these steps:

Clone this repository:
git clone https://github.com/EGZeelie/wave2midi.git
cd wave2midi

### Installation
   To install wave2midi, follow these steps
    
installation-steps

Clone this repository:
git clone https://github.com/your-username/wave2midi.git
cd wave2midi</code></pre>

Install the required dependencies:
pip install -r requirements.txt

Install the wave2midi package:
pip install -e 
       


Acknowledgments
Deezer for Spleeter source separation technology
The librosa development team for audio analysis tools
MIDI specification developers
Dependencies
Python 3.7+
librosa - for audio analysis
pydub - for audio processing
mido - for MIDI file creation
tensorflow or pytorch - for stem separation model
spleeter or similar source separation library
numpy, scipy - for numerical computations
Usage
Command Line Interface
# Basic usage
python wave2midi.py input.wav output_directory/

# With custom number of stems (5-7)
python wave2midi.py input.wav output_directory/ --stems 6

# With specific model and sample rate
python wave2midi.py input.wav output_directory/ --model 5stems --sample-rate 22050
Python API
from wave2midi import WaveToMIDIConverter

# Create converter instance
converter = WaveToMIDIConverter(stem_count=5)

# Convert WAV to MIDI stems
midi_files = converter.convert("input.wav", "output_dir/")

print(f"Created {len(midi_files)} MIDI files from stems")

## How It Works
Stem Separation: The input WAV file is processed through a source separation model that isolates 5-7 individual instrument/vocal tracks.
Audio Analysis: Each stem undergoes pitch detection, onset detection, and harmonic analysis to identify musical notes.
MIDI Conversion: The detected notes are converted to MIDI events with appropriate timing, velocity, and instrument information.
File Output: Each stem produces a separate MIDI file, allowing for individual editing and remixing.

## Configuration
The converter can be configured through a JSON configuration file or command-line arguments:

{
  "stem_count": 5,
  "sample_rate": 22050,
  "pitch_detection_method": "pyin",
  "min_note_duration": 0.1,
  "velocity_threshold": 0.3,
  "output_velocity_curve": "logarithmic",
  "model_weights": "pretrained/spleeter_5stems.h5"
}
Example Output
For an input file song.wav, the tool produces:

song_vocals.mid
song_drums.mid
song_bass.mid
song_piano.mid
song_other.mid
Limitations
Note This tool has some limitations to be aware of:

Complex polyphonic music may result in note detection inaccuracies
Very dense arrangements might not separate perfectly into distinct stems
Percussive elements may not translate perfectly to MIDI percussion tracks
Audio quality and recording conditions significantly affect conversion accuracy
Contributing
Contributions are welcome! Please read our contribution guidelines before submitting pull requests or opening issues.

License
This project is licensed under the MIT License - see the LICENSE file for details.
