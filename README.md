# Spark — Real-Time Twitter/X Sentiment Analysis 🚀

A distributed Natural Language Processing (NLP) pipeline built with Apache Spark to classify tweets as positive or negative. This project leverages the Kaggle "Sentiment140" dataset (1.6 million tweets) to demonstrate large-scale distributed computing principles using Spark's DataFrame API and MLlib.

## 🎯 Project Objectives
* Process 1.6 million records in parallel across a Spark cluster (simulated locally).
* Build a scalable Machine Learning pipeline including tokenization, TF-IDF vectorization, and Logistic Regression.
* Demonstrate core Parallel and Distributed Computing (PDC) concepts.
* Provide an interactive, real-time inference loop for custom text input.

---

## 🛠️ Prerequisites

Before running this project, ensure you have the following installed:

1. **Java 17 (Required for modern Spark compatibility)**
   * Ensure `JAVA_HOME` is set to your Java 17 installation path.
2. **Python 3.8 - 3.12**
   * *Note: Avoid highly experimental Python versions (e.g., 3.14) as PySpark relies on stable C-bindings.*
3. **Apache Spark & PySpark**
   * Install via pip: `pip install pyspark pandas`

---

## 🚀 Setup & Installation

**1. Clone the repository and set up a virtual environment:**
```bash
git clone https://github.com/HashirAhmadKamboh/Spark-Twitter-Sentiment-Analysis.git
cd PDC_Project
python -m venv spark-env
source spark-env/bin/activate
pip install pyspark pandas
```

**2. Download the Dataset:**
* Download the **Sentiment140 dataset** from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140).
* Extract the archive and place the `training.1600000.processed.noemoticon.csv` file into a `data/` directory at the root of this project.

*Directory Structure should look like this:*
```text
.
├── data/
│   └── training.1600000.processed.noemoticon.csv
├── pipeline.py
└── README.md
```

**3. Configure Environment Variables (Linux/Mac):**
Depending on your Java/Spark installation, you may need to pass JVM flags to bypass module encapsulation errors in Java 17+:
```bash
export PYSPARK_SUBMIT_ARGS="--driver-java-options '--add-exports java.base/sun.nio.ch=ALL-UNNAMED --add-opens java.base/java.lang=ALL-UNNAMED --add-opens java.base/java.lang.reflect=ALL-UNNAMED --add-opens java.base/java.io=ALL-UNNAMED --add-opens java.base/java.util=ALL-UNNAMED' pyspark-shell"
```

---

## 💻 Usage

Run the main pipeline script:
```bash
python pipeline.py
```

### What happens when you run it?
1. **Initial Run:** Spark ingests the 1.6M rows, cleans the text, trains the TF-IDF and Logistic Regression model, evaluates the accuracy (typically ~76%), and **saves the model to disk** in a `saved_sentiment_model/` directory.
2. **Subsequent Runs:** Spark detects the saved model and loads it directly in seconds, bypassing the training phase.
3. **Live Demo:** After evaluation, the terminal drops into an interactive loop where you can type custom sentences to get real-time sentiment predictions.

---

## 🧠 Parallel and Distributed Computing (PDC) Concepts Demonstrated

This project explicitly implements and demonstrates the following big-data paradigms:

* **Task Parallelism:** The `master("local[*]")` configuration allocates all available CPU cores as independent worker nodes.
* **Lazy Evaluation & DAGs:** Data ingestion and cleaning steps (narrow transformations) are deferred. Spark builds a Directed Acyclic Graph (DAG) and only executes computations when an Action (like `.count()` or `.fit()`) is called, optimizing memory usage.
* **Narrow vs. Wide Transformations:** 
  * *Narrow:* Regex cleaning and Tokenization happen entirely in-memory on individual partitions without network shuffling.
  * *Wide:* The IDF (Inverse Document Frequency) stage forces a cluster-wide shuffle to calculate global word frequencies across the entire dataset.
* **Distributed Machine Learning:** The Logistic Regression solver distributes gradient calculations across worker nodes, aggregating the loss updates back to the driver node to finalize the model weights.