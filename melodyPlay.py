from pydub import AudioSegment
import numpy as np
import simpleaudio as sa

# Define base note frequencies (A4 = 440 Hz)
BASE_FREQUENCIES = {
    'C': 261.63,
    'c': 277.18,  # C#
    'D': 293.66,
    'd': 311.13,  # D#
    'E': 329.63,
    'F': 349.23,
    'f': 369.99,  # F#
    'G': 392.00,
    'g': 415.30,  # G#
    'A': 440.00,
    'a': 466.16,  # A#
    'B': 493.88,
    'R': 0  # Rest
}

# Generate note frequencies for octaves 1-8
NOTE_FREQUENCIES = {
    f"{note}{octave}": freq * (2 ** (octave - 4))
    for octave in range(1, 9)
    for note, freq in BASE_FREQUENCIES.items() if freq > 0
}
NOTE_FREQUENCIES['R'] = 0  # Rest remains unchanged


# Generate a sine wave for a given frequency
def generate_sine_wave(frequency, duration_ms, sample_rate=44100, amplitude=0.5):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    wave = 0.5 * amplitude * np.sin(2 * np.pi * frequency * t)
    wave = (wave * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        wave.tobytes(), 
        frame_rate=sample_rate, 
        sample_width=wave.dtype.itemsize, 
        channels=1
    )
    return audio_segment

# Function to create a sequence of notes
def create_sequence(note_sequence, duration_ms=500):
    note_sequence = note_sequence.split()
    song = AudioSegment.silent(duration=0)
    for note in note_sequence:
        if note == 'R':  # Handle rest
            segment = AudioSegment.silent(duration=duration_ms)
        elif note in NOTE_FREQUENCIES:
            frequency = NOTE_FREQUENCIES[note]
            segment = generate_sine_wave(frequency, duration_ms)
        else:
            raise ValueError(f"Invalid note: {note}")
        song += segment
    return song


def play_melody(sequence, name, duration_ms=500):
    song = create_sequence(sequence, duration_ms)
    song.export(f"{name}.wav", format="wav")
    wave_obj = sa.WaveObject.from_wave_file(f"{name}.wav")

# Example sequence
# sequence = "C4 D4 E4 R G4 A4 F4 E7 D7"
# play_melody(sequence, "sample_melody")
