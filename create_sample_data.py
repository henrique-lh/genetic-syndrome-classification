import json
import numpy as np
import os

os.makedirs("sample_inputs", exist_ok=True)

EMBEDDING_DIM = 320
SYNDROME_CLASSES = [
    "100180860",
    "100192430",
    "100610443",
    "100610883",
    "300000007",
    "300000018",
    "300000034",
    "300000080",
    "300000082",
    "700018215",
]

np.random.seed(42)

for i in range(5):
    embedding = np.random.randn(EMBEDDING_DIM).astype(float)
    embedding = embedding / np.linalg.norm(embedding)

    sample_data = {
        "embedding": embedding.tolist(),
        "sample_id": f"sample_{i+1}",
    }

    filename = f"artifacts/sample_inputs/sample_{i+1}.json"
    with open(filename, "w") as f:
        json.dump(sample_data, f, indent=2)
    print(f"Created: {filename}")

batch_data = {
    "samples": [
        {
            "embedding": (
                np.random.randn(EMBEDDING_DIM)
                / np.linalg.norm(np.random.randn(EMBEDDING_DIM))
            ).tolist(),
            "sample_id": f"batch_sample_{i+1}",
        }
        for i in range(3)
    ]
}

batch_filename = "artifacts/sample_inputs/batch_input.json"
with open(batch_filename, "w") as f:
    json.dump(batch_data, f, indent=2)
print(f"Created: {batch_filename}")

print("\nSample data generation complete!")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"Possible syndrome classes: {SYNDROME_CLASSES}")
