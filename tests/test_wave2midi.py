import os
import pytest
import numpy as np
from scipy.io.wavfile import write as write_wav
from wave2midi import WaveToMIDIConverter

@pytest.fixture
def converter():
    """Fixture for a WaveToMIDIConverter instance with a mock separator."""
    # Mock the separator to avoid loading the real model
    class MockSeparator:
        def separate(self, waveform):
            # Return a dummy separation
            return {
                'vocals': np.zeros(len(waveform)),
                'other': waveform  # Pass the original waveform to 'other'
            }

    # Basic config for testing
    config = {
        "sample_rate": 22050
    }

    # Create converter instance
    converter = WaveToMIDIConverter(config)

    # Replace the separator with our mock
    converter.separator = MockSeparator()

    return converter

@pytest.fixture
def simple_wav_file(tmp_path):
    """Create a simple WAV file with a single sine tone for testing."""
    sample_rate = 22050
    duration = 1.0
    frequency = 440.0  # A4

    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

    wav_path = tmp_path / "test.wav"
    write_wav(wav_path, sample_rate, data)

    return wav_path, sample_rate, frequency

def test_detect_notes_sine_wave(converter, simple_wav_file):
    """Test note detection on a simple sine wave."""
    wav_path, sample_rate, frequency = simple_wav_file

    # Load the audio data
    import librosa
    audio, sr = librosa.load(wav_path, sr=sample_rate, mono=True)

    # Detect notes
    notes = converter.detect_notes(audio, sr)

    # Check that notes were detected
    assert len(notes) > 0, "No notes were detected"

    # Check that the detected pitch is correct (A4 = MIDI 69)
    detected_pitch = notes[0]['pitch']
    assert detected_pitch == 69, f"Expected MIDI pitch 69, but got {detected_pitch}"

    # Check that the duration is reasonable
    detected_duration = notes[0]['duration']
    assert 0.8 < detected_duration < 1.2, f"Expected duration around 1.0s, got {detected_duration}"

def test_convert_creates_midi_files(converter, simple_wav_file, tmp_path):
    """Test that the convert function creates MIDI files."""
    wav_path, _, _ = simple_wav_file
    output_dir = tmp_path / "output"

    # Mock the separate_stems method to return a predictable result
    def mock_separate_stems(wav_path):
        import librosa
        audio, sr = librosa.load(wav_path, sr=converter.config['sample_rate'], mono=True)

        return {
            'vocals': np.zeros_like(audio), # No vocals
            'drums': np.zeros_like(audio), # No drums
            'bass': np.zeros_like(audio), # No bass
            'other': audio # The original audio
        }

    converter.separate_stems = mock_separate_stems

    # Run the conversion
    midi_files = converter.convert(str(wav_path), str(output_dir))

    # Check that exactly one MIDI file was created (for the 'other' stem)
    assert len(midi_files) == 1, f"Expected 1 MIDI file, but {len(midi_files)} were created"

    # Check that the output directory was created
    assert os.path.isdir(output_dir), "Output directory was not created"

    # Check that the created MIDI file exists
    expected_midi_path = output_dir / "test_other.mid"
    assert os.path.exists(expected_midi_path), f"Expected MIDI file not found: {expected_midi_path}"

    # Check that no MIDI file was created for the silent stems
    for stem in ['vocals', 'drums', 'bass']:
        unexpected_midi_path = output_dir / f"test_{stem}.mid"
        assert not os.path.exists(unexpected_midi_path), f"MIDI file created for silent stem: {stem}"

def test_main_function_dry_run(mocker, tmp_path):
    """Test the main CLI function with a dry run mock."""
    # Mock the WaveToMIDIConverter to avoid actual processing
    mock_converter_instance = mocker.MagicMock()
    mock_converter_instance.convert.return_value = ["output/test.mid"]
    mock_converter = mocker.patch('wave2midi.WaveToMIDIConverter', return_value=mock_converter_instance)

    # Create a dummy input file
    input_wav = tmp_path / "input.wav"
    input_wav.touch()

    output_dir = tmp_path / "output"

    # Mock sys.argv
    mocker.patch('sys.argv', ['wave2midi.py', str(input_wav), str(output_dir)])

    # Run main
    from wave2midi import main
    main()

    # Assert that the converter was called with an empty config
    mock_converter.assert_called_once_with({})

    # Assert that convert was called
    mock_converter_instance.convert.assert_called_once_with(str(input_wav), str(output_dir))
