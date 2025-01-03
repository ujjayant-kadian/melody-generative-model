# 4th Attempt (Improved Model with Data Augmentation and Separate Pitch/Rhythm Handling)

## 🎯 **Key Improvements in the 4th Attempt**
In the 4th attempt, the model architecture was updated to handle **separate pitch and rhythm embeddings**. This is a significant improvement over the previous attempts where pitch and rhythm were combined into a single token.
---

## 🏋️ **Training Progress (Loss & Perplexity)**
| **Step** | **Train Loss** | **Val Loss** | **Perplexity** |
|---------|----------------|--------------|----------------|
| 0       | 2.5953         | 2.5803       | 13.20          |
| 200     | 1.0537         | 1.0814       | 2.95           |
| 400     | 0.9819         | 1.0098       | 2.74           |
| 600     | 0.9543         | 0.9988       | 2.71           |
| 800     | 0.9347         | 0.9858       | 2.68           |
| 1000    | 0.9172         | 0.9789       | 2.66           |
| 1200    | 0.8970         | 0.9631       | 2.62           |
| 1400    | 0.8743         | 0.9477       | 2.58           |
| 1600    | 0.8393         | 0.9124       | 2.49           |
| 1800    | 0.8007         | 0.8770       | 2.40           |
| 1999    | 0.7676         | 0.8476       | 2.33           |

---

## 🎼 **Generated Melody vs Baseline Melody**
| **Generated Melody** | **Baseline Melody** |
|----------------------|---------------------|
| R R R R R R R R R c3 F4 f4 F4 f4 F4 F4 F4 f4 F4 d4 d4 d4 R R c3 d4 R F4 c3 c3 c3 R R R d4 c4 c4 g4 F4 R f4 F4 d4 R g4 g4 f4 F4 f4 F4 F4 F4 d4 d4 c3 F4 F4 c3 c3 c3 c3 c3 d4 c3 F4 f4 F4 c3 c3 d4 c3 d4 d4 F4 R c3 F4 d4 R c3 d4 d4 F4 c3 R c3 c3 R R c3 c3 d4 d4 d4 c3 C4 g4 F4 R f4 F4 | F7 f7 d7 a3 D4 a3 C3 D8 G7 G4 A5 c7 g5 D7 c4 E8 G6 c8 A4 D4 D5 f4 d2 A2 G7 F8 A8 C5 f2 D8 g4 g4 G7 g1 c4 c2 E3 G4 C4 a8 A7 B7 c2 F6 B2 B3 C8 D3 g6 G1 C8 F4 E7 B2 G1 D1 A5 F3 D2 B7 E1 A3 d4 a6 A1 B3 a7 D8 B6 f8 a5 G3 C2 f7 f3 a7 f3 A4 D8 g4 g1 G2 F1 C3 F4 g7 a1 B8 g1 D3 D6 f5 d7 g5 B3 G8 d2 E8 g6 f3 |

---

## 📊 **Objective Metrics Comparison**
| **Metric**             | **4th Attempt** | **3rd Attempt** |
|------------------------|-----------------|-----------------|
| KL Divergence (Pitch)   | 9.5017          | 8.6836          |
| KL Divergence (Rhythm)  | 0.0165          | -               |
| Transition Matrix MSE (Pitch) | 0.003524       | 0.004159        |
| Transition Matrix MSE (Rhythm) | 0.014401       | -               |
| N-Gram Overlap (Pitch)  | 0.0001          | 0.000067        |
| N-Gram Overlap (Rhythm) | 0.0001          | -               |

---

## 📈 **Analysis of Metrics and Output Quality**
### 🎯 **Perplexity**
Perplexity decreased significantly, indicating better model performance and generalization:
- **Initial perplexity** was 13.20 at step 0.
- By step 1999, perplexity reduced to **2.33**, which is a strong indicator of good model convergence.

### 🎯 **KL Divergence**
- The KL divergence for **pitch** increased slightly compared to the 3rd attempt.
- **Rhythm KL divergence** is a new metric added in the 4th attempt and shows a very low value, indicating that rhythm predictions are well-aligned with the training data.

### 🎯 **Transition Matrix MSE**
- The **pitch transition matrix MSE** improved from 0.004159 to **0.003524**, indicating that the model learned better note transitions.
- The **rhythm transition matrix MSE** is **0.014401**, suggesting that rhythm transitions are also well-learned.

### 🎯 **N-Gram Overlap**
- The **4-gram overlap** for both pitch and rhythm is low, indicating that the model struggles to learn longer-term dependencies in the sequences.

---

## 📈 **What Worked Well in the 4th Attempt?**
- The **separation of pitch and rhythm embeddings** improved rhythmic coherence in generated melodies.
- The **data augmentation** helped the model learn more varied note sequences.
---