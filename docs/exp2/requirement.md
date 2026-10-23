
Project Task


Project Title: Deep Learning-Based Fine-Grained Feature Enhancement for Visual Pattern Recognition: Applications in Psychological Drawing Analysis and Sketch Understanding

Project Story
Imagine you visit a psychologist, and they hand you a blank sheet of paper and a pencil with a simple instruction:
•	“Draw a House, a Tree, and a Person.” (This is a classic psychological test called HTP).
•	Psychologists know that people reveal their inner emotional world through how they draw:
•	Someone feeling confident might draw a large house with open windows and a sturdy tree right in the center.
•	Someone struggling with severe depression or anxiety often draws very differently: maybe a tiny, fragile person tucked away in the bottom corner, faint strokes, or a house with no doors or windows.
The Big Problem
Right now, evaluating these drawings takes a lot of time from trained psychologists, and there simply aren't enough specialists.
Naturally, researchers tried using standard AI (computer vision) to help screen drawings automatically. But standard AI models have two big blind spots with drawings:
They lose tiny details: AI built for recognizing everyday photos (like cats, dogs, or cars) looks at big color patches. In a drawing, if a person draws a tiny, faint figure that takes up only 5% of the page, standard AI often blurs it away as "unimportant background noise." Yet in psychology, that tiny detail is the most important clinical clue.
Drawings are sparse lines, not colorful photos: A drawing is 95% blank white paper with thin pencil strokes connecting distant parts of the page. Traditional AI struggles to understand the relationship between a line drawn on the far left and a line drawn on the far right.
The Solution: The Core of This Dissertation
This dissertation creates AI models specifically designed to "see" and "read" psychological drawings the way a clinician does.
It accomplishes this through three main pillars:
1. Making Small Details Pop (SOANet)
The dissertation introduces a network that acts like a smart magnifying glass. Instead of letting tiny figures vanish, it deliberately zooms in on small objects (like a miniature person or a missing window) and gives them extra weight. It then translates these visual clues into psychological meaning (such as: "small size + bottom corner placement = feelings of withdrawal").

2. Reading the Whole Drawing Efficiently (The Mamba Model)
This is where Mamba comes in.
The old way (Transformers): Some modern AI models try to connect every single point on a drawing to every other point. This is like trying to compare every grain of sand on a beach—it takes massive amounts of computer memory and gets bogged down fast.
The Mamba way: Mamba is a cutting-edge architecture (a Selective State Space Model). Think of it like a reader scanning a line of text smoothly from left to right, up to down. It sweeps across the canvas in continuous paths, skipping the blank white spaces while remembering the important pencil strokes.
It connects the entire drawing together understanding how the tiny person relates to the distant house using a fraction of the computing power and memory of older AI models.
3. Combining Everything into a Reliable Tool (FASA-SOANet)
Finally, the project combines:
•	Fine details (pencil stroke pressure and faint lines),
•	Mamba-style global scanning (the overall layout), and
•	Psychological rules (guidelines from clinical psychology textbooks).
In One Sentence
This project builds an AI system that can scan a pencil drawing, clearly notice tiny, faint details that other computers miss, connect the whole layout together using the Mamba model, and give psychologists a fast, interpretable screening tool for depression.
Dataset to be used:
[Roboflow HTP Dataset]
          │
          ▼
 Stage 1: Train Object Detector (YOLOv8 / Faster R-CNN)
          │ (Learns to accurately locate House, Tree, Person & Small Details)
          ▼
 Frozen Detector applied to [DD Dataset]
          │
          ▼
 Automated Bounding Boxes & Area Ratios (Area < on total number of screen flagged as Small Object)
          │
          ▼
 Stage 2: SOANet / Mamba Sketch / FASA-SOANet
          │ (Trained on DD for Depression Classification & Semantic Feature Mapping)
          ▼
 Final Diagnostic Prediction + Interpretable Grad-CAM / Attention Maps


1. The DD set my supervisor sent by mail

•	Istanbul Bilgi University Dataset: Contains 450 HTP drawings from 83 child patients, with depression and anxiety scores annotated by psychologists [9]. We screen 78 samples containing small objects.

•	Zhang et al.’s 2023 Dataset: Includes 1615 HTP drawings (1296 normal, 319 depressed), with clinical diagnosis labels [15]. We select 132 samples with small objects.

•	Pan et al.’s 2022 Dataset: Consists of 3046 HTP drawings from college students, annotated with positive/negative psychological state labels [6]. We use this dataset for cross-validation.
•	The final dataset includes 210 samples with small objects (108 depressed, 102 normal), covering ages 4–25 years (53% male, 47% female).HTP Computer Vision Dataset

2. Link to Roboflow HTP-lewxu Dataset: HTP Computer Vision Dataset: https://universe.roboflow.com/htp/htp-lewxu
https://universe.roboflow.com/htp/htp-  lewxu/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true

Dataset simple


Key deliverables:

•	Phase 1: Data Preparation & Preprocessing (Crucial Foundation)
Dataset Splitting

•	Phase 2: Object Detection Pipeline (Using Roboflow Dataset)
     Train Detector:

•	Phase 3: Model Implementation
(SOANet), (Mamba Sketch) Benchmark Mamba against ResNet-50 and Vision Transformer (ViT), and CBAM-EfficientNet & FASA-SOANet

•	Phase 4: Validation & Interpretability
Comprehensive Metrics, Baseline vs. +CLAHE vs. +Small-Object Loss vs.   +CBAM/Mamba vs. Full Multimodal Fusion

Generate Grad-CAM/attention heatmaps across the test split to prove the networks activate on diagnostic elements

Cross-Dataset Generalization





THE END
