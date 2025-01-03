import random

# List of notes in order
NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
NOTE_SET = set(NOTES + ['R'])

def translate_notes(notes, shift):
    """
    Translate notes by a given shift (e.g., +1 for one semitone up).
    """
    notes = notes.split()  # Split input into individual notes
    translated_notes = []

    for note in notes:
        if note == 'R':  # Handle rests
            translated_notes.append(note)
            continue

        # Extract the base note and octave
        base_note = ''.join(filter(str.isalpha, note))
        octave = ''.join(filter(str.isdigit, note))

        if base_note in NOTES:
            index = NOTES.index(base_note)
            new_index = (index + shift) % len(NOTES)
            translated_note = NOTES[new_index] + octave
            translated_notes.append(translated_note)
        else:
            translated_notes.append(note)  # Keep unknown tokens as is

    return ' '.join(translated_notes)

# Example usage
# input_notes = "C4 D4 E4"
# shift = 1
# output_notes = translate_notes(input_notes, shift)
# print(output_notes)  # Output: "c4d4F4"

def string_to_notes(melody_str):
    return melody_str.split()

def notes_to_string(note_list):
    return ' '.join(note_list)

def random_local_pitch_variation(note_list, probability=0.01):
    """
    With a certain probability, shift note by ±1 semitone (mod 12).
    """
    new_notes = []
    for note in note_list:
        if note == 'R':
            new_notes.append(note)
            continue
        base_note = ''.join(filter(str.isalpha, note))
        octave = ''.join(filter(str.isdigit, note))
        if base_note in NOTES:
            if random.random() < probability:
                old_index = NOTES.index(base_note)
                shift = random.choice([-1, +1])
                new_index = (old_index + shift) % len(NOTES)
                varied_note = NOTES[new_index] + octave
                new_notes.append(varied_note)
            else:
                new_notes.append(note)  # Keep the note unchanged
        else:
            new_notes.append(note)  # Keep unknown tokens as is
    return new_notes
# Example usage
# note_list = ["C4", "D4", "E4", "R", "G4"]
# pitch_var_notes = random_local_pitch_variation(note_list, probability=0.5)
# print(pitch_var_notes)  # Output: ['C4', 'c4', 'E4', 'R', 'F4'] (randomized)

def invert_melody(note_list, reference_note='C4'):
    """
    Invert intervals around a chosen reference_note (e.g., 'C4').
    """
    reference_base = ''.join(filter(str.isalpha, reference_note))
    reference_octave = ''.join(filter(str.isdigit, reference_note))
    reference_octave = int(reference_octave) if reference_octave else 4
    if reference_base not in NOTES:
        reference_base = 'C'

    ref_idx = NOTES.index(reference_base)
    new_notes = []

    for note in note_list:
        if note == 'R':  # Handle rests
            new_notes.append(note)
            continue

        # Extract the base note and octave
        base_note = ''.join(filter(str.isalpha, note))
        octave = int(''.join(filter(str.isdigit, note)))

        if base_note in NOTES:
            idx = NOTES.index(base_note)
            # Distance from reference
            distance = idx - ref_idx
            # Invert the distance
            inverted_idx = (ref_idx - distance) % len(NOTES)
            inverted_base = NOTES[inverted_idx]

            # Adjust octave based on inversion
            octave_shift = (ref_idx - distance) // len(NOTES)
            inverted_octave = reference_octave + octave_shift

            new_notes.append(inverted_base + str(inverted_octave))
        else:
            new_notes.append(note)  # Keep unknown tokens as is

    return new_notes

# Example usage
# note_list = ["C4", "D4", "E4", "R", "G4"]
# inverted_notes = invert_melody(note_list, reference_note="C4")
# print(inverted_notes)  # Output: ['C4', 'B3', 'A3', 'R', 'F3']


def retrograde_melody(note_list):
    """
    Reverse the entire sequence, preserving rests as they appear in reverse order.
    """
    return note_list[::-1]
# Example usage
# note_list = ["C4", "D4", "E4", "R", "G4"]
# retrograde_notes = retrograde_melody(note_list)
# print(retrograde_notes)  # Output: ['G4', 'R', 'E4', 'D4', 'C4']

def insert_random_rests(note_list, probability=0.02):
    """
    Insert an extra rest after some notes with a low probability.
    """
    new_notes = []
    for note in note_list:
        new_notes.append(note)
        if note != 'R' and random.random() < probability:
            new_notes.append('R')
    return new_notes
# Example usage
# note_list = ["C4", "D4", "E4", "F4", "G4"]
# notes_with_rests = insert_random_rests(note_list, probability=0.5)
# print(notes_with_rests)  # Output: ['C4', 'R', 'D4', 'E4', 'R', 'F4', 'G4'] (randomized)

def swap_adjacent_notes(note_list, probability=0.005):
    """
    Randomly swap pairs of adjacent notes. 
    """
    i = 0
    while i < len(note_list) - 1:
        if random.random() < probability and note_list[i] != 'R' and note_list[i+1] != 'R':
            # swap
            note_list[i], note_list[i+1] = note_list[i+1], note_list[i]
            i += 2  # skip next
        else:
            i += 1
    return note_list
# Example usage
# note_list = ["C", "D", "E", "F", "G"]
# swapped_notes = swap_adjacent_notes(note_list, probability=0.5)
# print(swapped_notes)  # Output: ['D', 'C', 'E', 'F', 'G'] (randomized)

# Load the input file
with open('inputMelodies.txt', 'r') as file:
    input_melodies = file.readlines()

# Apply the different augmentation techniques and save

shifts = [1, 2, 3, 4, 5]
augmented_melodies = []

for melody in input_melodies:
    melody = melody.strip()


    # Convert to a list of tokens
    note_list = string_to_notes(melody)
    
    # Append original simplified melody
    augmented_melodies.append(notes_to_string(note_list))

    # 1)
    for shift in shifts:
        shifted_notes = translate_notes(melody, shift)  # your existing function
        augmented_melodies.append(shifted_notes)
    
    # 2) Additional advanced transformations
    
    # Local pitch variation
    pitch_var_notes = random_local_pitch_variation(note_list)
    # Insert rests
    # pitch_var_notes = insert_random_rests(pitch_var_notes, probability=0.02)
    pitch_var_str = notes_to_string(pitch_var_notes)
    augmented_melodies.append(pitch_var_str)
    
    # Inversion
    inverted_notes = invert_melody(note_list, reference_note='C4')
    inv_str = notes_to_string(inverted_notes)
    augmented_melodies.append(inv_str)
    
    # Retrograde
    retro_notes = retrograde_melody(note_list)
    retro_str = notes_to_string(retro_notes)
    augmented_melodies.append(retro_str)


# Save the augmented melodies to a new file
with open('inputMelodiesAugmented.txt', 'w') as file:
    for melody in augmented_melodies:
        file.write(melody + '\n')

print("The augmented melodies have been saved to inputMelodiesAugmented.txt")


