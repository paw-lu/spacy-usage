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

# Chapter 1: Finding words, phrases, names, and concepts


## Introduction to spaCy


Start by creating the nlp object
based on language.
It will include language-specific rules
for processes such as tokenization.


### NLP object

```python
# Import English language class
import spacy.lang.en

# Create nlp object
nlp = spacy.lang.en.English()
```

### Doc object

```python
# Process a string of text with nlp object
doc = nlp("Hello world!")

# Iterate over tokens in a Doc
for token in doc:
    print(token.text)
```

### Token object

```python
doc = nlp("Hello world!")

# Index into the Doc to get a single token
token = doc[1]

# Get the token text via the ``.text`` attribute
print(token.text)
```

### Span object

```python
doc = nlp("Hello world")

# A slice from Doc is a Span
span = doc[1:3]

# Get the span text via ``.text`` attribute
print(span.text)
```

### Lexical attributes

```python
doc = nlp("It costs $5")

print("Index:   ", [token.i for token in doc])
print("Text:    ", [token.text for token in doc])

print("is_alpha:", [token.is_alpha for token in doc])
print("is_punct:", [token.is_punct for token in doc])
print("like_num:", [token.like_num for token in doc])
```

## Getting started


### English

```python
# Import the English language class
from spacy.lang.en import English

# Create the nlp object
nlp = English()

# Process a text
doc = nlp("This is a sentence.")

# Print the document text
print(doc.text)
```

### German

```python
# Import the German language class
from spacy.lang.de import German

# Create the nlp object
nlp = German()

# Process a text (this is German for: "Kind regards!")
doc = nlp("Liebe Grüße!")

# Print the document text
print(doc.text)
```

### Spanish

```python
# Import the Spanish language class
from spacy.lang.es import Spanish

# Create the nlp object
nlp = Spanish()

# Process a text (this is Spanish for: "How are you?")
doc = nlp("¿Cómo estás?")

# Print the document text
print(doc.text)
```

## Documents, spans, and tokens


When you call `nlp` on a string,
spaCy tokenizes and creates a document object.

- **`doc.text`**
  to get the token as a string

```python
# Import the English language class and create the nlp object
from spacy.lang.en import English

nlp = English()

# Process the text
doc = nlp("I like tree kangaroos and narwhals.")

# Select the first token
first_token = doc[0]

# Print the first token's text
print(first_token.text)
```

```python
# Import the English language class and create the nlp object
from spacy.lang.en import English

nlp = English()

# Process the text
doc = nlp("I like tree kangaroos and narwhals.")

# A slice of the Doc for "tree kangaroos"
tree_kangaroos = doc[2:4]
print(tree_kangaroos.text)

# A slice of the Doc for "tree kangaroos and narwhals" (without the ".")
tree_kangaroos_and_narwhals = doc[2:6]
print(tree_kangaroos_and_narwhals.text)
```

## Lexial attributes


Use `Doc` and `Token` objects
along with lexical attributes
to find percentages in text.

- `like_num`
  to check whether a token resembles a number.

- `token.i + 1`
   to find the token following the current token

```python
from spacy.lang.en import English

nlp = English()

# Process the text
doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

# Iterate over the tokens in the doc
for token in doc:
    # Check if the token resembles a number
    if token.like_num:
        # Get the next token in the document
        next_token = doc[token.i + 1]
        # Check if the nex token's text equals "%"
        if next_token.text == "%":
            print("Percentage found:", token.text)

```

## Statistical models


### What are statistical models?

Statistical models enable spaCy to predict linguistic attributes in context.

Trained on labeled example texts.
Can be updates with more examples to fine-tune.

<!-- #region -->
### Model packages

```shell
python -m spacy download en_core_web_sm
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")
```
<!-- #endregion -->

### Predicting part-of-speech tags

```python
import spacy

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Process a text
doc = nlp("She ate the pizza")

# Iterate over the tokens
for token in doc:
    # Print the text and the predicted part-of-speech tag
    print(token.text, token.pos_)
```

### Predicting syntatic dependencies

```python
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)
```

### Dependency label scheme


```text
She ate the pizza
```

| Label     | Description          | Example |
| --------- | -------------------- | ------- |
| **nsubj** | nominal subject      | She     |
| **dobj**  | direct object        | pizza   |
| **det**   | determiner (article) | the     |



### Predicting named entities


`Apple `**`ORG`** is looking at buying `U.K. ``**GPE**` startup for `$1 billion `**`MONEY`**

```python
# Process a text
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its labels
    print(ent.text, ent.label_)
```

### `spacy.explain`

Get quick definitions
of most common tags and labels.

```python
spacy.explain("GPE")
```

```python
spacy.explain("NNP")
```

```python
spacy.explain("dobj")
```

```python
spacy.explain("MONEY")
```

## Model packages


Model packages do not include
the labeled data that the model was trained on.


## Loading Models

- **`python -m spacy download en_core_web_sm`** to download models
- **`spacy.load`** to load them

```python
import spacy

# Load the "en_core_web_sm" model
nlp = spacy.load("en_core_web_sm")

text = (
    "It’s official:"
    " Apple is the first U.S. public company "
    " to reach a $1 trillion market value"
)
# Process the text
doc = nlp(text)

# Print the document text
print(doc.text)
```

## Predicting linguistic annotation

- **`spacy.explain`**
  to find out what a tag or label means
- **`token.pos_`**
  to return part-of-speech tag
- **`token.dep_`**
  to return dependency label
- **`doc.ents`**
  to return predicted entities
- **`doc.ents.label_`**
  to return entity label

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = (
    "It’s official:"
    " Apple is the first U.S. public company"
    " to reach a $1 trillion market value"
)

# Process the text
doc = nlp(text)

for token in doc:
    # Get the token text, part-of-speech tag, and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    print(f"{token_text:<12}{token_pos:<10}{token_dep:<10}")
```

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = (
    "It’s official:"
    " Apple is the first U.S. public company"
    " to reach a $1 trillion market value"
)

doc = nlp(text)

# Iterate over the predicted entities
for ent in doc.ents:
    # Print the entity text and its label
    print(ent.text, ent.label_)
```

## Predicting named entities in context

The model is often wrong.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Upcoming iPhone X release date leaked as Apple reveals pre-orders"

# Process the text
doc = nlp(text)

# Iterate over the entities
for ent in doc.ents:
    # Print the entity text and label
    print(ent.text, ent.label_)

# Get the span for "iPhone X"
iphone_x = doc[1:3]

# Print the span text
print("Missing entity:", iphone_x.text)
```

## Rule-based matching


### Why not regular expressions?

- Match on `Doc` object,
  not just strings
- Match on token and token attributes
- Use the model's predictions
- Distinguish between "duck" (verb) and "duck" (noun)


### Math patterns

- List of dictionaries—
  one per token
- Match exact token text
  
  ```python
  [{"TEXT": "iPhone"}, {"TEXT": "X"}]
  ```

- Match lexical attributes

  ```python
  [{"LOWER": "iphone"}, {"TEXT": "x"}]
  ```
  
- Match any token attribute

  ```python
  [{"LEMMA": "buy"}, {"POS": "NOUN"}]
  ```


### Using the matcher

```python
import spacy

# Import the Matcher
from spacy.matcher import Matcher

# Load the model and create the nlp object
nlp = spacy.load("en_core_web_sm")

# Initialize the matcher with the shared vocab
matcher = Matcher(nlp.vocab)

# Add the pattern to the matcher
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", None, pattern)

# Process text
doc = nlp("Upcoming iPhone X release date leaked")

# Call the matcher on the doc
matches = matcher(doc)

# Iterate over the matcher
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
```

### Matching lexical attributes

```python
pattern = [
    {"IS_DIGIT": True},
    {"LOWER": "fifa"},
    {"LOWER": "world"},
    {"LOWER": "cup"},
    {"IS_PUNCT": True}
]
doc = nlp("2018 FIFA World Cup: France won!")

matcher = Matcher(nlp.vocab)
matcher.add("FIFA_PATTERN", None, pattern)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
```

### Matching other token attributes


```python
pattern = [{"LEMMA": "love", "POS": "VERB"}, {"POS": "NOUN"}]
doc = nlp("I loved dogs but now I love cats more.")
matcher = Matcher(nlp.vocab)
matcher.add("FIFA_PATTERN", None, pattern)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
```

### Using operators and quantifiers

```python
pattern = [
    {"LEMMA": "buy"},
    {"POS": "DET", "OP": "?"},  # optional: match 0 or 1 times
    {"POS": "NOUN"},
]
doc = nlp("I bought a smartphone. Now I'm buying apps.")
matcher = Matcher(nlp.vocab)
matcher.add("FIFA_PATTERN", None, pattern)
matches = matcher(doc)
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)
```

| Example        | Description                  |
| -------------- | ---------------------------- |
| `{"OP": "!"}`  | Negation: match 0 times      |
| `{"OP": "?"}`  | Optional: match 0 or 1 times |
| `{"OP": "+"}`  | Match 1 or more times        |
| `{"OP": "\*"}` | Match 0 or more times        |



## Using the matcher


- **`Matcher`**
  match phrases to patterns,
  initialize with vocabulary
- **`nlp.vocab`**
  list out vocab of model
- **`matcher.add`**
  add pattern to the matcher

```python
import spacy

# Import the matcher
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
doc = nlp("Upcoming iPhone X release date leaked as Apple reveals pre-orders")

# Initialize the Matcher with the shared vocabulary
matcher = Matcher(nlp.vocab)

# Create a pattern matching two tokens--"iPhone" and "X"
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]

# Add the pattern to the matcher
matcher.add("IPHONE_X_PATTERN", None, pattern)

# Use hte matcher on the doc
matches = matcher(doc)
print("Matches:", [doc[start:end].text for match_id, start, end in matches])
```

### Writing match patterns

Write more complex match patterns
using different token attributes and operators

Write one pattern
that only matches mentions of the _full_ iOS versions—
"iOS 7", "iOS 11", and "iOS 10".

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

# Write a pattern for full iOS versions ("iOS 7", "iOS 11", and "iOS 10")
pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("IOS_VERSION_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)
```

Now write _one_ pattern
that only matches forms of "download"—
tokens with lemma "download"—
followed by a token with the part-of-speech tag `"PROPN"`
(proper noun).

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

doc = nlp(
    "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

# Write a pattern that matches a form of "download" plus proper noun
pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("DOWNLOAD_THINGS_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)
```

Write _one_ pattern
that matches adjectives (`"ADJ"`)
followed by one or two `"NOUN"`s—
one noun and one optional noun.

```python
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

# Write a pattern for adjective plus one or two nouns
pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]

# Add the pattern to the matcher and apply the matcher to the doc
matcher.add("ADJ_NOUN_PATTERN", None, pattern)
matches = matcher(doc)
print("Total matches found:", len(matches))

# Iterate over the matches and print the span text
for match_id, start, end in matches:
    print("Match found:", doc[start:end].text)
```

```python

```
