import os
from mido import MidiFile, MidiTrack, Message

# Map MIDI note numbers to note names with octave information
MIDI_NOTE_TO_NAME_WITH_OCTAVE = {
    midi_note: f"{note[0].lower() if '#' in note else note}{midi_note // 12 - 1}"
    for midi_note, note in enumerate([
        'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] * 11)
}

NAME_WITH_OCTAVE_TO_MIDI_NOTE = {
    v: k for k, v in MIDI_NOTE_TO_NAME_WITH_OCTAVE.items()
}

def midi_to_text_sequence_with_octaves(midi_path):
    midi = MidiFile(midi_path)
    sequence = []
    last_note_time = 0

    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_with_octave = MIDI_NOTE_TO_NAME_WITH_OCTAVE.get(msg.note, '')
                if note_with_octave:
                    # Add rests for the duration between the last note and the current note
                    if msg.time > last_note_time:
                        rest_duration = msg.time - last_note_time
                        num_rests = rest_duration // 480  # Assuming 480 ticks per beat
                        sequence.extend(['R '] * num_rests)
                    sequence.append(note_with_octave)
                    sequence.append(' ')
                    last_note_time = msg.time

    # Remove consecutive rests
    sequence = [note for i, note in enumerate(sequence) if not (note == 'R' and i > 0 and sequence[i - 1] == 'R')]

    return ''.join(sequence)

def text_sequence_to_midi(sequence, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    sequence = sequence.strip().split()
    time = 0

    for item in sequence:
        if item == 'R':
            # Rest corresponds to a delay in time
            time += 480  # Assuming 480 ticks per beat for each rest
        else:
            note = NAME_WITH_OCTAVE_TO_MIDI_NOTE.get(item)
            if note is not None:
                if time > 0:
                    track.append(Message('note_off', note=note, velocity=0, time=time))
                track.append(Message('note_on', note=note, velocity=64, time=0))
                time = 480  # Duration of the note (1 beat)

    # Add a final note_off message to close the last note
    if time > 0:
        track.append(Message('note_off', note=note, velocity=0, time=time))

    midi.save(output_path)

midi_dir = 'musicDatasetOriginal'

output_dir = 'musicDatasetSimplified'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

text_sequences = []

for file_name in os.listdir(midi_dir):
    if file_name.endswith('.mid'):
        midi_path = os.path.join(midi_dir, file_name)
        text_sequence = midi_to_text_sequence_with_octaves(midi_path)
        if text_sequence:  # Check if the sequence is not empty
            text_sequences.append(text_sequence)
        else:
            print(f"No notes found in {file_name}")  # Debugging output

with open("inputMelodies.txt", "w") as file:
    for sequence in text_sequences:
        file.write(sequence + "\n")
        
for i, sequence in enumerate(text_sequences):
    output_path = os.path.join(output_dir, f"output_midi_{i+1}.mid")
    text_sequence_to_midi(sequence, output_path)

print("Text sequences with octaves have been written to inputMelodies.txt")
