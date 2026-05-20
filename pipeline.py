import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

def main():
    print("--- Step 1: Initializing Spark Session ---")
    spark = SparkSession.builder \
        .appName("TwitterX-Sentiment-Analysis") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")

    print("\n--- Step 2: Ingesting Sentiment140 Dataset ---")
    schema = StructType([
        StructField("target", IntegerType(), True),
        StructField("id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("flag", StringType(), True),
        StructField("user", StringType(), True),
        StructField("text", StringType(), True)
    ])

    data_path = "data/training.1600000.processed.noemoticon.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        spark.stop()
        return

    raw_df = spark.read.csv(data_path, schema=schema, header=False)
    
    print("\n--- Step 3: Data Cleaning & Preprocessing ---")
    cleaned_df = raw_df.select(
        F.when(F.col("target") == 4, 1).otherwise(0).alias("label"),
        F.col("text")
    ).withColumn(
        "clean_text", F.regexp_replace(F.col("text"), r"@[A-Za-z0-9_]+", "")
    ).withColumn(
        "clean_text", F.regexp_replace(F.col("clean_text"), r"https?://[A-Za-z0-9./]+", "")
    ).withColumn(
        "clean_text", F.regexp_replace(F.col("clean_text"), r"[^a-zA-Z0-9\s]", "")
    ).withColumn(
        "clean_text", F.lower(F.trim(F.col("clean_text")))
    ).filter(
        F.col("clean_text") != ""
    )

    print("\n--- Step 4: Building the Distributed ML Pipeline ---")
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10)

    # THIS is the line that was missing in your error!
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

    print("\n--- Step 5: Training or Loading the Model ---")
    model_path = "saved_sentiment_model"
    train_data, test_data = cleaned_df.randomSplit([0.8, 0.2], seed=42)

    if os.path.exists(model_path):
        print(f"-> Found existing model at '{model_path}'. Loading it directly...")
        model = PipelineModel.load(model_path)
        print("-> Model loaded successfully!")
    else:
        print("-> No saved model found. Training from scratch. This will take a few minutes...")
        model = pipeline.fit(train_data)
        print("-> Model training complete!")
        model.save(model_path)
        print(f"-> Model saved to directory: {model_path}")

    print("\n--- Step 6: Making Predictions on Test Data ---")
    predictions = model.transform(test_data)
    predictions.select("clean_text", "label", "prediction", "probability").show(5, truncate=50)

    print("\n--- Step 7: Evaluating Model Performance ---")
    multi_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = multi_evaluator.evaluate(predictions)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    binary_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = binary_evaluator.evaluate(predictions)
    print(f"Area Under ROC (AUC): {auc:.4f}")

    print("\n--- Step 8: Live Demo (Interactive Inference) ---")
    print("Type a sentence to see how the model classifies it!")
    
    while True:
        try:
            user_input = input("\nEnter text to test (or type 'exit' to quit): ")
        except EOFError:
            break
            
        if user_input.lower() in ['exit', 'quit']:
            break
            
        test_df = spark.createDataFrame([(user_input,)], ["raw_text"])
        
        clean_test_df = test_df.withColumn(
            "clean_text", F.regexp_replace(F.col("raw_text"), r"@[A-Za-z0-9_]+", "")
        ).withColumn(
            "clean_text", F.regexp_replace(F.col("clean_text"), r"https?://[A-Za-z0-9./]+", "")
        ).withColumn(
            "clean_text", F.regexp_replace(F.col("clean_text"), r"[^a-zA-Z0-9\s]", "")
        ).withColumn(
            "clean_text", F.lower(F.trim(F.col("clean_text")))
        )
        
        user_pred = model.transform(clean_test_df)
        result = user_pred.select("prediction", "probability").first()
        
        if result and clean_test_df.first()["clean_text"] != "":
            sentiment = "Positive 😊" if result.prediction == 1.0 else "Negative 😞"
            confidence = result.probability[1] if result.prediction == 1.0 else result.probability[0]
            print(f"-> Predicted Sentiment: {sentiment} (Confidence: {confidence * 100:.2f}%)")
        else:
            print("-> Text was completely filtered out (e.g., only contained special characters).")

    spark.stop()

if __name__ == "__main__":
    main()