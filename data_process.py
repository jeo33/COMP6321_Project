import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse, vstack as sp_vstack
from data_loader import RCV1DataLoader



RANDOM_SEED = 42

def compute_class_counts(y):
    if issparse(y):
        counts = np.asarray(y.sum(axis=0)).ravel()
    else:
        counts = y.sum(axis=0)
    return counts.astype(int)


def plot_label_distribution(y, title, save_path):
    counts = compute_class_counts(y)
    sorted_counts = np.sort(counts)[::-1]

    plt.figure(figsize=(16, 5))
    xs = np.arange(len(sorted_counts))
    plt.bar(xs, sorted_counts, color=plt.cm.viridis(
        np.linspace(0, 1, len(sorted_counts))))
    plt.yscale("log")
    plt.xlabel("Class Index (sorted by frequency)", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot: {save_path}")

MIN_COUNT_DROP = 500

def balance_by_dropping_rare_classes(X, y, metadata, min_count=MIN_COUNT_DROP):
    counts = compute_class_counts(y)
    keep_classes = np.where(counts >= min_count)[0]

    print(f"\n[Drop] Total classes: {y.shape[1]}, keep {len(keep_classes)} "
          f"with count >= {min_count}")

    y_reduced = y[:, keep_classes]

    if issparse(y_reduced):
        doc_has_label = np.asarray(y_reduced.sum(axis=1) > 0).ravel()
    else:
        doc_has_label = (y_reduced.sum(axis=1) > 0)

    X_bal = X[doc_has_label]
    y_bal = y_reduced[doc_has_label]

    print(f"[Drop] Kept samples: {X_bal.shape[0]} / {X.shape[0]}")

    new_metadata = dict(metadata)
    new_metadata["n_train"] = int(X_bal.shape[0])
    new_metadata["n_categories"] = int(y_bal.shape[1])
    # 只保留对应的 target_names
    target_names = metadata["target_names"]
    new_metadata["target_names"] = [target_names[i] for i in keep_classes]
    new_metadata["balance_strategy"] = f"drop_classes_min_count_{min_count}"

    return X_bal, y_bal, new_metadata

OVERSAMPLE_TARGET_PCT = 75

def balance_by_oversampling(X, y, metadata, target_pct=OVERSAMPLE_TARGET_PCT,
                            random_seed=RANDOM_SEED):
    rng = np.random.default_rng(random_seed)

    counts = compute_class_counts(y)
    target_count = int(np.percentile(counts, target_pct))
    print(f"\n[Over] Target count per class = {target_count} "
          f"(percentile {target_pct})")

    n_samples, n_classes = y.shape
    base_indices = np.arange(n_samples)

    extra_indices = []

    for c in range(n_classes):
        c_count = counts[c]
        if c_count >= target_count:
            continue

        need = target_count - c_count

        if issparse(y):
            col = y[:, c].toarray().ravel()
            docs_with_c = base_indices[col > 0.5]
        else:
            docs_with_c = base_indices[y[:, c] > 0.5]

        if len(docs_with_c) == 0:
            continue

        sampled = rng.choice(docs_with_c, size=need, replace=True)
        extra_indices.append(sampled)

    if extra_indices:
        extra_indices = np.concatenate(extra_indices)
    else:
        extra_indices = np.array([], dtype=int)

    all_indices = np.concatenate([base_indices, extra_indices])

    if issparse(X):
        X_bal = sp_vstack([X, X[extra_indices]], format=X.format)
    else:
        X_bal = X[all_indices]

    y_bal = y[all_indices]

    print(f"[Over] Original samples: {n_samples}, after oversampling: "
          f"{X_bal.shape[0]}")

    new_metadata = dict(metadata)
    new_metadata["n_train"] = int(X_bal.shape[0])
    new_metadata["balance_strategy"] = (
        f"oversample_to_percentile_{target_pct}_count_{target_count}"
    )

    return X_bal, y_bal, new_metadata


def save_balanced_dataset(data_dir_out, X_train, y_train,
                          X_val, y_val, X_test, y_test, metadata):
    os.makedirs(data_dir_out, exist_ok=True)

    with open(os.path.join(data_dir_out, "train_data.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(data_dir_out, "train_labels.pkl"), "wb") as f:
        pickle.dump(y_train, f)

    with open(os.path.join(data_dir_out, "val_data.pkl"), "wb") as f:
        pickle.dump(X_val, f)
    with open(os.path.join(data_dir_out, "val_labels.pkl"), "wb") as f:
        pickle.dump(y_val, f)

    with open(os.path.join(data_dir_out, "test_data.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    with open(os.path.join(data_dir_out, "test_labels.pkl"), "wb") as f:
        pickle.dump(y_test, f)

    with open(os.path.join(data_dir_out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved balanced dataset to: {data_dir_out}")



def main():
    # 原始数据
    base_loader = RCV1DataLoader(data_dir="data")
    X_train, y_train = base_loader.load_data("train")
    X_val, y_val = base_loader.load_data("val")
    X_test, y_test = base_loader.load_data("test")
    metadata = base_loader.load_metadata()


    X_train_drop, y_train_drop, meta_drop = balance_by_dropping_rare_classes(
        X_train, y_train, metadata, min_count=MIN_COUNT_DROP
    )

    save_balanced_dataset(
        data_dir_out=os.path.join("data", "balanced_drop"),
        X_train=X_train_drop,
        y_train=y_train_drop,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metadata=meta_drop,
    )

    plot_label_distribution(
        y_train_drop,
        title=f"Train Distribution after Dropping (min_count={MIN_COUNT_DROP})",
        save_path="results/plots/train_distribution_drop.png",
    )

    X_train_over, y_train_over, meta_over = balance_by_oversampling(
        X_train, y_train, metadata, target_pct=OVERSAMPLE_TARGET_PCT,
        random_seed=RANDOM_SEED
    )

    save_balanced_dataset(
        data_dir_out=os.path.join("data", "balanced_oversample"),
        X_train=X_train_over,
        y_train=y_train_over,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        metadata=meta_over,
    )

    plot_label_distribution(
        y_train_over,
        title=("Train Distribution after Oversampling "
               f"(target={OVERSAMPLE_TARGET_PCT}th percentile)"),
        save_path="results/plots/train_distribution_oversample.png",
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
