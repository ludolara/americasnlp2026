# AmericasNLP 2026 Shared Task: Cultural Image Captioning for Indigenous Languages

The AmericasNLP 2026 Shared Task challenges participants to develop systems that generate accurate, culturally grounded captions for images depicting Indigenous cultures of the Americas, written in the Indigenous languages themselves.

**Update (22th of April): We have updated the Guarani test set to remove duplicate entries. Please download the revised version and regenerate your predictions. If you submit the old test set, we will keep only the first prediction for each duplicated item as your official entry!**

**Update (20th of April): The test set has been released!**

**Update (13th of April): We have released a surprise addition to our language lineup: Orizaba Nahuatl (nlv), part of the broader Nahuatl (nah) language family!**

## Motivation

Many Indigenous languages of the Americas are endangered and lack the resources needed to train NLP systems effectively. Language communities are actively pursuing revitalization, but creating culturally grounded teaching materials is expensive and time-consuming. Image captioning systems present an opportunity to generate such materials at scale, but doing so requires not only linguistic competence but also cultural knowledge — understanding the people, traditions, and contexts depicted in the images.

## Task Description

Participants are given a dataset of culturally situated images, each paired with a caption in the associated Indigenous language. The goal is to generate captions for unseen images.

**Example:**

| | |
|---|---|
| **Image** | [A wooden structure](data/pilot/images/wixarika/hch_009.jpg) |
| **Target Caption (Wixárika)** | *Ik+ kareta m+ya kaxetuni wixárika wapait+ yu +kú puti utá, uti xainék+ metá tsiere manapait+ rá ye hupú.* |
| **English** | The so-called carretón, built specifically to store food like corn, is also used as housing for people. |

## Rules

### General

- Participants may use the provided training and development data, plus **any additional resources** (external data, pretrained models, etc.).
- Participants must **not** create test outputs manually or train on the test sets ~~or train on the development~~ (**UPDATE:** participants are allowed to use the development set for training)

### Submission Format

We will provide participants with a JSONL file (alongside the images) containing the following fields (same as dev set, but without the `target_caption`: `id`, `filename`, `split`, `culture`, `language`, and `iso_lang`. Participants must generate a `predicted_caption` for each entry and include it as an additional field in their submission file.

### How to Submit

Send your submission via email to americas.nlp.workshop@gmail.com

In the email body, include:

- Line 1: Team name
- Line 2: Names of all team members
- Line 3: All languages you are sending submissions for, in the order of your choice (we will use this to double-check that we received all the files you intended to send)
[optional]
- Line 4: A link to a GitHub repository with code that can be used to reproduce your results. This is not required to participate in the shared task, but it is strongly encouraged.


**Attachment:**

Please attach all output files to your email as a single zip file, named after your team (e.g., `TeamName.zip`). Within that zip file, individual files should be named `<language>-<version>.jsonl` (e.g., `bribri-0.jsonl`). The language name should match the one used in the corresponding evaluation set. The version number lets you submit multiple runs of your system per language; it must be a single digit (please don't submit more than 9 versions per language).


### Evaluation

We adopt a **two-stage evaluation protocol**:
1. **Stage 1:** All systems are ranked using **ChrF++**.
2. **Stage 2:** The top-5 systems are evaluated by **human judges** according to a fixed set of criteria.

Participants can enter for as many languages as they like; each language is evaluated separately. We provide an evaluation script and a baseline system to help get started.

## Languages

| Language | Region |
|---|---|
| Bribri | Costa Rica |
| Guaraní | Paraguay |
| Yucatec Maya | Mexico |
| Wixárika | Mexico |
| Orizaba Nahuatl (nlv), part of broader Nahuatl (nah) | Mexico |

## Data

### Pilot

Pilot data is available under [`data/pilot/`](data/pilot/). Each dataset is provided as a JSONL file with corresponding images. See [`data/pilot/wixarika.jsonl`](data/pilot/wixarika.jsonl) for an example.

> **⚠️ Note:** The pilot data includes Spanish captions for reference, but these are provided **only in the pilot set**. Spanish captions will **not** be included in the development or test sets and should not be relied upon for building systems.

### Development

Development data is available under [`data/dev/`](data/dev/) for Bribri, Guaraní, Maya, Wixárika and Nahuatl. Each language folder contains a JSONL file and corresponding images.

## Important Dates

| Date | Milestone |
|---|---|
| ~~February 20, 2026~~ | ~~Release of pilot data and baseline system~~ |
| ~~March 1, 2026~~ | ~~Release of development sets (50 examples)~~ |
| ~~April 1, 2026~~ ~~April 13, 2026~~ | ~~Release of surprise languages~~ |
| ~~April 20, 2026~~ | ~~Release of test sets~~ |
| May 1, 2026 | Submission of results (shared task deadline) |
| May 8, 2026 | Winner announcement |
| May 13, 2026 | Submission of system description paper |
| May 15, 2026 | Acceptance notification for system description papers |
| May 22, 2026 | Camera-ready version due |

All deadlines are **11:59 pm UTC-12h (AoE)**.

## Registration

If you are interested in participating, please register here: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSe1OPQzxCRWDMBbKi_PTuKsSSnVWB_5PAcs8HsZYhxoImO4BQ/viewform?usp=header)


## Contact

[americas.nlp.workshop@gmail.com](mailto:americas.nlp.workshop@gmail.com)
