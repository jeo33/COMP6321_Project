from datasets import load_dataset
try:
    print("Attempting to load rcv1...")
    dataset = load_dataset("rcv1", split="train", trust_remote_code=True)
    print("Success!")
    print(dataset[0])
except Exception as e:
    print(f"Failed: {e}")
