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

# 3: Processing pipelines

Notes and examples
on spaCy's processing pipelines.
Covers what goes on under the hood
when you process text,
how to write your own components
and add them to the pipeline,
and how to use custom attributes
to add your own metadata to documents, spans and tokens.


## Processing pipelines


### Built-in pipeline components

| Name    | Description             | Creates                                           |
| ------- | ----------------------- | ------------------------------------------------- |
| tagger  | Part-of-speech tagger   | Token.tag, Token.pos                              |
| parser  | Dependency parser       | Token.dep, Token.head, Doc.sents, Doc.noun_chunks |
| ner     | Named entity recognizer | Doc.ents, Token.ent_iob, Token.ent_type           |
| textcat | Text classifier         | Doc.cats                                          |



### Under the hood

Pipeline is defined in model's `meta.json`.
Built-in components need binary data to make predictions.


### Pipeline attributes

- **`nlp.pipe_names`**
  list of pipeline component names
- **`nlp.pipeline`**
  list of `(name, component)` tuples

```python
import spacy

nlp = spacy.load("en_core_web_sm")
print(nlp.pipe_names)
print(nlp.pipeline)
```

<!-- #region -->
## What happens when you call nlp?

When a user calls

```python
doc = nlp("This is a sentence")
```

spaCy tokenizes the text
and applies each pipeline component in order.
<!-- #endregion -->

## Inspecting the pipeline

Inspect the small English model's pipeline

- **`nlp.pipe_names`**
  list of pipeline component names
- **`nlp.pipeline`**
  list of `(name, component)` tuples

```python
import spacy

# Load the en_core_web_sm model
nlp = spacy.load("en_core_web_sm")

# Print the names of the pipeline components
print(nlp.pipe_names)

# Print the full pipeline of (name, component) tuples
print(nlp.pipeline)
```

## Custom pipeline components


### Why custom components?

Can make a function automatically execute
when `nlp` is called.
Can add custom metadata to documents and tokens.
Can update built-in attributes such as `doc.ents`

<!-- #region -->
### Anatomy of a component

Function that takes a `doc`,
modifies it,
and returns it.
Can be added using `nlp.add_pipe`

```python
def custom_component(doc):
    # Do something to the doc here
    return doc
nlp.add_pipe(custom_component)
```

| Argument | Description          | Example                                 |
| -------- | -------------------- | --------------------------------------- |
| last     | If True, add last    | nlp.add_pipe(component, last=True)      |
| first    | If True, add first   | nlp.add_pipe(component, first=True)     |
| before   | Add before component | nlp.add_pipe(component, before="ner")   |
| after    | Add after component  | nlp.add_pipe(component, after="tagger") |

<!-- #endregion -->

### Example: a simple component

```python
# Create the nlp object
nlp = spacy.load("en_core_web_sm")

# Define a custom component
def custom_component(doc):
    # Print the doc's length
    print("Doc length:", len(doc))
    # Return the doc object
    return doc


# Add the component first in the pipeline
nlp.add_pipe(custom_component, first=True)

# Print the pipeline component names
print("Pipeline:", nlp.pipe_names)
```

```python
# Create the nlp object
nlp = spacy.load("en_core_web_sm")

# Define a custom component
def custom_component(doc):

    # Print the doc's length
    print("Doc length:", len(doc))

    # Return the doc object
    return doc


# Add the component first in the pipeline
nlp.add_pipe(custom_component, first=True)

# Process a text
doc = nlp("Hello world!")
```

## Use cases for custom components

Custom components can be used to

- Compute your own values based on tokens and their attributes
- Add named entities,
  such as dictionary based ones


## Simple components

A simple component
that prints the length of a document.

- **`nlp.add_pipe`**
  add custom component to pipeline

```python
import spacy

# Define a custom component
def length_component(doc):
    # Get the doc's length
    doc_length = len(doc)
    print(f"This document is {doc_length} tokens long.")
    # Return the doc
    return doc

# Load the small English model
nlp = spacy.load("en_core_web_sm")

# Add the component first in the pipeline and print the pipe names
nlp.add_pipe(length_component, first=True)
print(nlp.pipe_names)

# Process a text
doc = nlp("This is a sentence.")
```

## Complex components

Create a custom component
that uses `PhraseMatcher`
to find animal names
in the document,
and adds the matched spans to `doc.ents`.

Add it _after_ the `ner` component.

```python
import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
animal_patterns = list(nlp.pipe(animals))
print("animal_patterns:", animal_patterns)
matcher = PhraseMatcher(nlp.vocab)
matcher.add("ANIMAL", None, *animal_patterns)

# Define the custom component
def animal_component(doc):
    # Apply the matcher to the doc
    matches = matcher(doc)
    # Create a Span for each match and assign the label "ANIMAL"
    spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
    # Overwrite the doc.ents with the matched spans
    doc.ents = spans
    return doc


# Add the component to the pipeline after the "ner" component
nlp.add_pipe(animal_component, after="ner")
print(nlp.pipe_names)

# Process the text and print the text and label for the doc.ents
doc = nlp("I have a cat and a Golden Retriever")
print([(ent.text, ent.label_) for ent in doc.ents])
```

```python

```
