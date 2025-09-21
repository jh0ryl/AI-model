import pandas as pd
import random

# Load your original corpus
corpus_df = pd.read_csv("corpus.csv")

# Add a new column "citations" with random values between 0 and 1000
corpus_df["citations"] = [random.randint(0, 1000) for _ in range(len(corpus_df))]

# Save the updated file
corpus_df.to_csv("corpus_with_citations.csv", index=False)

print("✅ New file saved as corpus_with_citations.csv with a 'citations' column.")
