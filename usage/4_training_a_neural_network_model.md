---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: spacy
    language: python
    name: spacy
---

# 4: Training a neural network model


Customize spaCy's models for specific cases,
such as predict a new entity type in online comments.

<!-- #region -->
## Training and updating models

### Why update a model?

- Better results on your specific domain
- Learn classification schemes specifically for your problem
- Essential for text classification
- Very useful for named entity recognition
- Less critical for part-of-speech tagging
  and dependency parsing
  
### How training works

1. **Initialize** model weights randomly with `nlp.begin_training`
2. **Predict** a few examples with the current weights by calling `nlp.update`
3. **Compare** prediction with true labels
4. **Calculate** how to change weights to improve predictions
5. **Update** weights slightly
6. **Repeat** from step 2

![Training diagram](images/training.png)

### Training the entity recognizer

The entity recognizer
tags words and phrases in context.
Each token can only belong
to one entity.
Examples need to come
with context.

```python
("iPhone X is coming", {"entities": [(0, 8, "GADGET")]})
```

Texts with no entities are also important.

```python
("iPhone X is coming", {"entities": [(0, 8, "GADGET")]})
```

The goal is to teach the model to generalize.

### The training data

Examples of what we want to predict in context.
Updating an existing model
requires a few hundred to a few thousand examples.
Training a new category
requires a few thousand to a million examples.

Usually created manually be human annotators.
Can be semi-automated using `Matcher`.
<!-- #endregion -->

## Purpose of training

You almost always want to fine-tune
spaCy's pre-trained models
with more examples.
This will not help you
discover patterns in unlabeled data though.


## Creating training data

- **`Matcher`**
  can be used
  to quickly create training data
  for named entity models

Want to find all mentions
of different iPhone models.

Create training data
to teach a model
to recognize them as `"GADGET"`.

1. Write a pattern
   for two tokens
   whose lowercase forms
   match `"iphone"` and `"x"`.
2. Write a pattern
   for two tokens—
   one whose lowercase form
   matches `"iphone"`
   and a digit
   using the `"?"` operator

```python
import json
from spacy.matcher import Matcher
from spacy.lang.en import English

TEXTS = [
    "How to preorder the iPhone X",
    "iPhone X is coming",
    "Should I pay $1,000 for the iPhone X?",
    "The iPhone 8 reviews are here",
    "iPhone 11 vs iPhone 8: What's the difference?",
    "I need a new phone! Any tips?",
]

nlp = English()
matcher = Matcher(nlp.vocab)

# Two tokens whose lowercase forms match "iphone" and "x"
pattern1 = [{"LOWER": "iphone"}, {"LOWER": "x"}]

# Token whose lwoercase form matches "iphone" and a digit
pattern2 = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]

# Add patterns to the matcher and check the result
matcher.add("GADGET", None, pattern1, pattern2)
for doc in nlp.pipe(TEXTS):
    print([doc[start:end] for match_id, start, end in matcher(doc)])
```

Use the match patterns created
to make a set of training examples.

1. Create a doc object for each text using `nlp.pipe`
2. Math on the `doc` to create a list of matching spans
3. Format each example as a tuple
   of the text and a dict
4. Append to training data

```python
import json

from spacy.matcher import Matcher
from spacy.lang.en import English

TEXTS = [
    "How to preorder the iPhone X",
    "iPhone X is coming",
    "Should I pay $1,000 for the iPhone X?",
    "The iPhone 8 reviews are here",
    "iPhone 11 vs iPhone 8: What's the difference?",
    "I need a new phone! Any tips?",
]

print(TEXTS)

nlp = English()
matcher = Matcher(nlp.vocab)
pattern1 = [{"LOWER": "iphone"}, {"LOWER": "x"}]
pattern2 = [{"LOWER": "iphone"}, {"IS_DIGIT": True}]
matcher.add("GADGET", None, pattern1, pattern2)

TRAINING_DATA = []

# Create a doc object for each text in TEXTS
for doc in nlp.pipe(TEXTS):
    # Match on ther doc and create a list of matches spans
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    # Get (start character, end character, label) tuples of matcher
    entities = [(span.start_char, span.end_char, "GADGET") for span in spans]
    # Format the matches as a (doc.text, entities) tuple
    training_example = (doc.text, {"entities": entities})
    # Append the example to the training data
    TRAINING_DATA.append(training_example)

print(*TRAINING_DATA, sep="\n")
```

<!-- #region -->
## The training loop

### Steps

1. **Loop**
   for a number of times.
2. **Shuffle**
   the training data.
3. **Divide**
   the model for each batch
4. **Save**
   the updated model

### How training works

![spaCy training](images/training.png)

- **Training data:**
  Examples and their annotations
- **Text:**
  The input text
  the model should predict a label for
- **Gradient:**
  How to change the weights
  
### Example loop

```python
TRAINING_DATA = [
    ("How to preorder the iPhone X", {"entities": [(20, 28, "GADGET")]})
    # And many more examples...
]

# Loop for 10 iterations
for i in range(10):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    # Create batches and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA):
        # Split the batch in texts and annotations
        texts = [text for text, annotation in batch]
        annotations = [annotation for text, annotation in batch]
        # Update the model
        nlp.update(texts, annotations)

# Save the model
nlp.to_disk(path_to_model)
```

### Update an existing model

- Improve the predictions
  on new data
- Especially useful
  to improve existing categories—
  like `"PERSON"`
- Can add new categories
- Take care
  to prevent model does not "forget"
  old categories
  
### Setting up a new pipeline from scratch

```python
# Start with blank English model
nlp = spacy.blank("en")
# Create blank entity recognizer and add it to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)
# Add a new label
ner.add_label("GADGET")

# Start the training
nlp.begin_training()
# Train for 10 iterations
for itn in range(10):
    random.shuffle(examples)
    # Divide examples into batches
    for batch in spacy.util.minibatch(examples, size=2):
        texts = [text for text, annotation in batch]
        annotations = [annotation for text, annotation in batch]
        # Update the model
        nlp.update(texts, annotations)
```
<!-- #endregion -->

## Setting up a pipeline

Prepare a pipeline
to train the entity recognizer
to recognize `"GADGET"`.

- **`spacy.blank`**
  Create a blank model

```python
import spacy

# Create a blank "en" model
nlp = spacy.blank("en")

# Create a new entity recognizer and add it to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)

# Add the label "GADGET" to the entity recognizer
ner.add_label("GADGET")
```

## Building a training loop

- **`nlp.begin_training`**
  create a training loop
- **`spacy.util.minibatch`**
  create batches
  of training data
- **`nlp.update`**
  update the model
  with texts and annotations

```python
import json
import random

import spacy

# Training data generated from before
TRAINING_DATA = [
    ["How to preorder the iPhone X", {"entities": [[20, 28, "GADGET"]]}],
    ["iPhone X is coming", {"entities": [[0, 8, "GADGET"]]}],
    ["Should I pay $1,000 for the iPhone X?", {"entities": [[28, 36, "GADGET"]]}],
    ["The iPhone 8 reviews are here", {"entities": [[4, 12, "GADGET"]]}],
    ["Your iPhone goes up to 11 today", {"entities": [[5, 11, "GADGET"]]}],
    ["I need a new phone! Any tips?", {"entities": []}],
]

# Work from previous cell
nlp = spacy.blank("en")
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)
ner.add_label("GADGET")

# Start the training
nlp.begin_training()

# Loop for 10 iterations
for itn in range(10):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}

    # Batch the examples and iterate over them
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities, in batch]
        annotations = [entities for text, entities in batch]

        # Update the model
        nlp.update(texts, annotations, losses=losses)
    print(losses)
```

<!-- #region -->
## Exploring the model

These are the model results

| Text                                                                                                              | Entities             |
| ----------------------------------------------------------------------------------------------------------------- | -------------------- |
| Apple is slowing down the iPhone 8 and iPhone X - how to stop it                                                  | (iPhone 8, iPhone X) |
| I finally understand what the iPhone X ‘notch’ is for                                                             | (iPhone X,)          |
| Everything you need to know about the Samsung Galaxy S9                                                           | (Samsung Galaxy,)    |
| Looking to compare iPad models? Here’s how the 2018 lineup stacks up                                              | (iPad,)              |
| The iPhone 8 and iPhone 8 Plus are smartphones designed, developed, and marketed by Apple                         | (iPhone 8, iPhone 8) |
| what is the cheapest ipad, especially ipad pro???                                                                 | (ipad, ipad)         |
| Samsung Galaxy is a series of mobile computing devices designed, manufactured and marketed by Samsung Electronics | (Samsung Galaxy,)    |


The model is 70% correct—
it missed suffixes such as
`"Pro"`, `"S9"`, and `"Plus"`.
<!-- #endregion -->

<!-- #region -->
## Training best practices

### Problem 1: Models can forget

Existing model can overfit on new data.
If you only update it with `"WEBSITE"`
it can "unlearn" what `"PERSON"` is.

### Solution 1: Mix in previously corrected predictions

If you're training `"WEBSITE"`
include examples of `"PERSON"`.

```python
# Bad
TRAINING_DATA = [
    ("Reddit is a website", {"entities": [(0, 6, "WEBSITE")]})
]

# Good
TRAINING_DATA = [
    ("Reddit is a website", {"entities": [(0, 6, "WEBSITE")]}),
    ("Obama is a person", {"entities": [(0, 5, "PERSON")]})
]
```

### Problem 2: Models can't learn everything

spaCy's models make predictions
on local context.
Model can struggle to learn
if decision is difficult to make
based on context.
Label scheme
needs to be consistent
and not too specific.

For example,
`"CLOTHING"` is better than `"ADULT_CLOTHING"` and `"CHILDRENS_CLOTHING"`.

### Solution 2: Plan your label scheme carefully

Pick categories
that are reflected
in local context.
More generic
is better than
too specific.
User rules
to go from generic labels
to specific categories.

```python
# Bad
LABELS = ["ADULT_SHOES", "CHILDRENS_SHOES", "BANDS_I_LIKE"]

# Good
LABELS = ["CLOTHING", "BAND"]
```
<!-- #endregion -->

<!-- #region -->
## Good data vs bad data

```python
TRAINING_DATA = [
    (
        "i went to amsterdem last year and the canals were beautiful",
        {"entities": [(10, 19, "TOURIST_DESTINATION")]},
    ),
    (
        "You should visit Paris once in your life, but the Eiffel Tower is kinda boring",
        {"entities": [(17, 22, "TOURIST_DESTINATION")]},
    ),
    ("There's also a Paris in Arkansas, lol", {"entities": []}),
    (
        "Berlin is perfect for summer holiday: lots of parks, great nightlife, cheap beer!",
        {"entities": [(0, 6, "TOURIST_DESTINATION")]},
    ),
]
```

This training data
is problematic
because a tourist destination
is a subjective judgement
and not a definitive category.

A better approach
would be to only label `"GPE"` or `"LOCATION"`,
then use a rule-based system
to determine whether entity
is a tourist destination—
like looking them up
in a travel wiki.

A fixed version
would look like this:
<!-- #endregion -->

```python
TRAINING_DATA = [
    (
        "i went to amsterdem last year and the canals were beautiful",
        {"entities": [(10, 19, "GPE")]},
    ),
    (
        "You should visit Paris once in your life, but the Eiffel Tower is kinda boring",
        {"entities": [(17, 22, "GPE")]},
    ),
    (
        "There's also a Paris in Arkansas, lol",
        {"entities": [(15, 20, "GPE"), (24, 32, "GPE")]},
    ),
    (
        "Berlin is perfect for summer holiday: lots of parks, great nightlife, cheap beer!",
        {"entities": [(0, 6, "GPE")]},
    ),
]
```

<!-- #region -->
## Training multiple labels

```python
TRAINING_DATA = [
    (
        "Reddit partners with Patreon to help creators build communities",
        {"entities": [(0, 6, "WEBSITE"), (21, 28, "WEBSITE")]},
    ),
    ("PewDiePie smashes YouTube record", {"entities": [(18, 25, "WEBSITE")]}),
    (
        "Reddit founder Alexis Ohanian gave away two Metallica tickets to fans",
        {"entities": [(0, 6, "WEBSITE")]},
    ),
    # And so on...
]
```

If a model trained on this data
is doing great on `"WEBSITE"`,
but doesn't recognize `"PERSON"` anymore,
it's likely because `"PERSON"` entities
occur in the training data,
but are not labeled,
so the model learned
that this label is incorrect.

A better version looks like:

```python
TRAINING_DATA = [
TRAINING_DATA = [
    (
        "Reddit partners with Patreon to help creators build communities",
        {"entities": [(0, 6, "WEBSITE"), (21, 28, "WEBSITE")]},
    ),
    (
        "PewDiePie smashes YouTube record",
        {"entities": [(0, 9, "PERSON"), (18, 25, "WEBSITE")]},
    ),
    (
        "Reddit founder Alexis Ohanian gave away two Metallica tickets to fans",
        {"entities": [(0, 6, "WEBSITE"), (15, 29, "PERSON")]},
    ),
    # And so on...
]
]
```
<!-- #endregion -->

## Conclusion

- Extract linguistic features—
  part-of-speech tags,
  dependencies,
  and named entities
- Use pre-trained statistical models
- Find words and phrases
  using `Matcher`
  and `PhraseMatcher` match rules
- Best practices
  for working with data structures—
  `Doc`, `Token`, `Span`, `Vocab`, and `Lexeme`.
- Use word vectors to find semantic similarities
- Scale up spaCey pipelines
  and improve performance
- Create training data
- Train and update
  spaCy's neural network models
  with new data

### More things to do

- [Training and updating other pipeline components](https://spacy.io/usage/training)
  - Part-of-speech tagger
  - Dependency parser
  - Text classifier
- [Customize the tokenizer](https://spacy.io/usage/linguistic-features#tokenization)
  - Adding rules and exceptions
    to split text differently

```python

```
