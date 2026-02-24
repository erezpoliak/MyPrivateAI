# SciRAG-QA: Multi-domain Closed-Question Benchmark Dataset for Scientific QA

This dataset contains question-answer (QA) pairs extracted from scientific papers across various areas of research. Each QA pair is linked to a specific paper title and area of research. The dataset is structured as follows:

```
/SciRAG-QA
  ├── README.md
  ├── dataset.json
  ├── dataset.csv
  └── metadata.csv
```

## Format

### `metadata.csv`

The `metadata.csv` file provides detailed meta-information regarding the papers used to extract the questions. This file contains the following fields:

- **`index`**: A 4-code index encoding area, sub-area, topic, and count per topic.
- **`Area`**: One of the 10 defined research areas.
- **`Sub-area`**: The specific sub-area within the research area (e.g., Physics → Electronics).
- **`Topic`**: The research topic (trending, emerging, or popular) within the sub-area.
- **`Title`**: The title of the paper.
- **`DOI`**: The DOI to access the paper.
- **`Authors`**: A list containing the names of the authors.
- **`Date`**: The publication date (can be the year, the year and month, or the full date).
- **`Venue`**: The journal title or conference where the paper was published.
- **`Publisher`**: The name of the publisher.

### `dataset.json`

The dataset is provided in JSON format, with an optional `dataset.csv` file that contains the same information in CSV format. Each entry in the dataset represents a QA pair with the following fields:

- **`ID`**: A 32-character long identifier for the QA pair.
- **`Question`**: The question.
- **`Answer`**: The answer.
- **`Type`**: The type of the answer. Possible values are:
  - `Binary`
  - `Float`
  - `Integer`
  - `N/A`
  - `Percentage`
  - `Scientific Notation`
  - `Text (Multi-word)`
  - `Text (Single-word)`
- **`Complexity`**: The difficulty level of the question, indicating how challenging it is to answer based on how it is referenced in the text (ranging from 1 to 4, with 4 being the most difficult).
- **`Source_IDX`**: The paper `index`, which references the corresponding entry in `metadata.csv`.
- **`Gold_REF`**: The exact part of the paper used to generate the QA pair.

## Example

```json
{
    "ID": "a693f1023ea657e1e093b8c3cd437941",
    "Question": "Are changes in farmland habitats under R75 expected to benefit biodiversity more than under R0?",
    "Answer": "Yes",
    "Type": "Binary",
    "Complexity": 2,
    "Source_IDX": "[2][1][0][3]",
    "Gold_REF": "Inference of: significant increase in moderate and high-suitability habitats under R75 with the negative trends in suitability under R0"
}
```

This example illustrates how each QA pair is structured, providing clear references to the source paper and the rationale for the answer.
