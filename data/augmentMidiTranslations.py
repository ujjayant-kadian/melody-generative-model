# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:06:10 2024

@author: Giovanni Di Liberto
See description in the assignment instructions.
"""
import random

# Define the note dictionary
NOTE_FREQUENCIES = {
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
}

# List of notes in order
NOTES = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
NOTE_SET = set(NOTES + ['R'])

def translate_notes(notes, shift):
    """
    Translate notes by a given shift (e.g., +1 for one semitone up).
    """
    notes = notes.replace(' ', '')
    translated_notes = []
    for note in notes:
        if note in NOTES:
            index = NOTES.index(note)
            new_index = (index + shift) % len(NOTES)
            translated_notes.append(NOTES[new_index])
        else:
            translated_notes.append(note)  # Keep the character as is if it's not a note
    return ''.join(translated_notes)
# Example usage
#input_notes = "CDE"
#shift = 1
#output_notes = translate_notes(input_notes, shift)
#print(output_notes)  # Output: cdF

def string_to_notes(melody_str):
    return melody_str.split()

def notes_to_string(note_list):
    return ''.join(note_list)

def random_local_pitch_variation(note_list, probability=0.05):
    """
    With a certain probability, shift note by ±1 semitone (mod 12).
    """
    new_notes = []
    for note in note_list:
        if note == 'R' or note not in NOTE_SET:
            new_notes.append(note)
            continue
        if random.random() < probability:
            old_index = NOTES.index(note)
            shift = random.choice([-1, +1])
            new_index = (old_index + shift) % len(NOTES)
            new_notes.append(NOTES[new_index])
        else:
            new_notes.append(note)
    return new_notes
# Example usage
# note_list = ["C", "D", "E", "R", "G"]
# pitch_var_notes = random_local_pitch_variation(note_list, probability=0.5)
# print(pitch_var_notes)  # Output: ['C', 'c', 'E', 'R', 'F'] (randomized)

def invert_melody(note_list, reference_note='C'):
    """
    Invert intervals around a chosen reference_note (e.g., 'C').
    """
    if reference_note not in NOTES:
        reference_note = 'C'
    ref_idx = NOTES.index(reference_note)
    new_notes = []
    for note in note_list:
        if note == 'R' or note not in NOTE_SET:
            new_notes.append(note)
        else:
            idx = NOTES.index(note)
            # Distance from reference
            distance = idx - ref_idx
            # Invert the distance
            inverted_idx = (ref_idx - distance) % len(NOTES)
            new_notes.append(NOTES[inverted_idx])
    return new_notes
# Example usage
# note_list = ["C", "D", "E", "R", "G"]
# inverted_notes = invert_melody(note_list, reference_note="C")
# print(inverted_notes)  # Output: ['C', 'B', 'A', 'R', 'F']


def retrograde_melody(note_list):
    """
    Reverse the entire sequence, preserving rests as they appear in reverse order.
    """
    return note_list[::-1]
# Example usage
# note_list = ["C", "D", "E", "R", "G"]
# retrograde_notes = retrograde_melody(note_list)
# print(retrograde_notes)  # Output: ['G', 'R', 'E', 'D', 'C']

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
# note_list = ["C", "D", "E", "F", "G"]
# notes_with_rests = insert_random_rests(note_list, probability=0.5)
# print(notes_with_rests)  # Output: ['C', 'R', 'D', 'E', 'R', 'F', 'G'] (randomized)

def swap_adjacent_notes(note_list, probability=0.02):
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
    
    # 1)
    for shift in shifts:
        shifted_notes = translate_notes(melody, shift)  # your existing function
        augmented_melodies.append(shifted_notes)
    
    # 2) Additional advanced transformations
    # (Example: We'll do random local pitch variation + insertion of rests)
    
    # a) Local pitch variation
    pitch_var_notes = random_local_pitch_variation(note_list, probability=0.05)
    # b) Insert rests
    pitch_var_notes = insert_random_rests(pitch_var_notes, probability=0.02)
    pitch_var_str = notes_to_string(pitch_var_notes)
    augmented_melodies.append(pitch_var_str)
    
    # d) Inversion
    inverted_notes = invert_melody(note_list, reference_note='C')
    inv_str = notes_to_string(inverted_notes)
    augmented_melodies.append(inv_str)
    
    # e) Retrograde
    retro_notes = retrograde_melody(note_list)
    retro_str = notes_to_string(retro_notes)
    augmented_melodies.append(retro_str)


# Save the augmented melodies to a new file
with open('inputMelodiesAugmented.txt', 'w') as file:
    for melody in augmented_melodies:
        file.write(melody + '\n')

print("The augmented melodies have been saved to inputMelodiesAugmented.txt")


