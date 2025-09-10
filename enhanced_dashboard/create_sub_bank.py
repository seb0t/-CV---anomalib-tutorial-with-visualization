#!/usr/bin/env python3
"""
Sub-Bank Creator from Embeddings Bank

Creates a sub-bank by clustering embeddings and sampling balanced representatives.
Features:
- Automatic optimal cluster detection using Silhouette Score
- Balanced sampling (5% of embeddings)
- Progress bars for all operations
- Robust error handling

Usage:
    python create_sub_bank.py [--sample-ratio 0.05] [--max-clusters 50]

Arguments:
    --sample-ratio: Percentage of embeddings to sample (default: 0.05 = 5%)
    --max-clusters: Maximum number of clusters to test (default: 50)
"""

import sys
import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Add the enhanced_dashboard directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def load_embeddings_bank(filepath: str = "embeddings_bank.pkl") -> np.ndarray:
    """Load embeddings bank from disk."""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def save_sub_bank(sub_bank: np.ndarray, n_clusters: int = 10, filepath: str = None):
    """Save sub-bank to disk with cluster info in filename."""
    if filepath is None:
        filepath = f"sub_bank_k{n_clusters}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(sub_bank, f)


def load_sub_bank(n_clusters: int = 10, filepath: str = None) -> np.ndarray:
    """Load sub-bank from disk based on cluster count."""
    if filepath is None:
        filepath = f"sub_bank_k{n_clusters}.pkl"
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def find_optimal_clusters(embeddings, max_clusters=50, min_clusters=2):
    """Find optimal number of clusters using Silhouette Score."""
    
    n_samples = len(embeddings)
    max_clusters = min(max_clusters, n_samples // 2)  # Can't have more clusters than half the samples
    
    if max_clusters < min_clusters:
        print(f"⚠️  Not enough data for clustering. Using {min_clusters} clusters.")
        return min_clusters
    
    print(f"🔍 Finding optimal number of clusters (testing {min_clusters} to {max_clusters})...")
    
    silhouette_scores = []
    k_range = range(min_clusters, max_clusters + 1)
    
    # Test different numbers of clusters
    with tqdm(k_range, desc="Testing cluster counts", unit="k") as pbar:
        for k in pbar:
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score
            score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(score)
            
            pbar.set_postfix({'k': k, 'silhouette': f'{score:.3f}'})
    
    # Find optimal k
    optimal_idx = np.argmax(silhouette_scores)
    optimal_k = k_range[optimal_idx]
    optimal_score = silhouette_scores[optimal_idx]
    
    print(f"🎯 Optimal number of clusters: {optimal_k} (silhouette score: {optimal_score:.3f})")
    
    return optimal_k


def create_balanced_sub_bank(embeddings, n_clusters, sample_ratio=0.05):
    """Create a balanced sub-bank by clustering and sampling."""
    
    print(f"🧠 Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    
    with tqdm(total=1, desc="Fitting K-means") as pbar:
        cluster_labels = kmeans.fit_predict(embeddings)
        pbar.update(1)
    
    # Calculate samples per cluster
    total_samples = int(len(embeddings) * sample_ratio)
    samples_per_cluster = max(1, total_samples // n_clusters)
    
    print(f"📊 Sampling strategy:")
    print(f"   • Total embeddings: {len(embeddings):,}")
    print(f"   • Sample ratio: {sample_ratio*100:.1f}%")
    print(f"   • Target samples: {total_samples:,}")
    print(f"   • Samples per cluster: {samples_per_cluster}")
    
    # Sample from each cluster
    sub_bank_embeddings = []
    cluster_info = []
    
    with tqdm(total=n_clusters, desc="Sampling from clusters", unit="cluster") as pbar:
        for cluster_id in range(n_clusters):
            # Get embeddings for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_size = len(cluster_embeddings)
            
            if cluster_size == 0:
                pbar.set_postfix({'cluster': cluster_id, 'size': 0, 'sampled': 0})
                pbar.update(1)
                continue
            
            # Sample from this cluster
            n_samples = min(samples_per_cluster, cluster_size)
            
            if n_samples == cluster_size:
                # Take all embeddings if cluster is small
                sampled_embeddings = cluster_embeddings
            else:
                # Random sampling
                indices = np.random.choice(cluster_size, n_samples, replace=False)
                sampled_embeddings = cluster_embeddings[indices]
            
            sub_bank_embeddings.extend(sampled_embeddings)
            cluster_info.append({
                'cluster_id': cluster_id,
                'total_size': cluster_size,
                'sampled': len(sampled_embeddings)
            })
            
            pbar.set_postfix({
                'cluster': cluster_id,
                'size': cluster_size,
                'sampled': len(sampled_embeddings)
            })
            pbar.update(1)
    
    # Convert to numpy array
    sub_bank = np.array(sub_bank_embeddings)
    
    # Print cluster statistics
    print(f"\n📈 Cluster Statistics:")
    for info in cluster_info:
        ratio = info['sampled'] / info['total_size'] * 100 if info['total_size'] > 0 else 0
        print(f"   Cluster {info['cluster_id']:2d}: {info['total_size']:5d} → {info['sampled']:4d} ({ratio:5.1f}%)")
    
    return sub_bank, n_clusters


def main():
    parser = argparse.ArgumentParser(description="Create sub-bank from embeddings bank")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.05,
        help="Percentage of embeddings to sample (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=50,
        help="Maximum number of clusters to test (default: 50)"
    )
    parser.add_argument(
        "--force-clusters",
        type=int,
        help="Force specific number of clusters (skip optimization)"
    )

    args = parser.parse_args()

    print("🏦 Sub-Bank Creator")
    print("=" * 50)
    print(f"📊 Sample ratio: {args.sample_ratio*100:.1f}%")
    print(f"🔍 Max clusters to test: {args.max_clusters}")
    if args.force_clusters:
        print(f"🎯 Forced clusters: {args.force_clusters}")
    print()

    # Load embeddings bank
    print("📂 Loading embeddings bank...")
    embeddings_bank = load_embeddings_bank("enhanced_dashboard/embeddings_bank.pkl")

    if embeddings_bank is None:
        print("❌ Error: Could not load embeddings_bank.pkl")
        print("💡 Make sure to run create_bank.py first to create the embeddings bank.")
        return 1

    print(f"✅ Loaded embeddings bank with shape: {embeddings_bank.shape}")
    print(f"📊 Total embeddings: {len(embeddings_bank):,}")
    print(f"🎯 Embedding dimension: {embeddings_bank.shape[1]}")
    print()

    # Find optimal number of clusters or use forced value
    if args.force_clusters:
        optimal_k = args.force_clusters
        print(f"🎯 Using forced number of clusters: {optimal_k}")
    else:
        optimal_k = find_optimal_clusters(embeddings_bank, args.max_clusters)

    print()

    # Create sub-bank
    try:
        sub_bank, n_clusters = create_balanced_sub_bank(
            embeddings_bank, 
            optimal_k, 
            args.sample_ratio
        )

        print(f"\n✅ Sub-bank created successfully!")
        print(f"📊 Final sub-bank shape: {sub_bank.shape}")
        print(f"🗂️  Total sub-bank embeddings: {len(sub_bank):,}")
        print(f"📉 Reduction: {len(embeddings_bank):,} → {len(sub_bank):,} ({len(sub_bank)/len(embeddings_bank)*100:.1f}%)")

        # Save sub-bank
        print(f"\n💾 Saving sub-bank...")
        save_sub_bank(sub_bank, n_clusters)
        print(f"✅ Sub-bank saved as: sub_bank_k{n_clusters}.pkl")

        # Verify save
        print("\n🔍 Verifying save...")
        from functions.helpers import load_sub_bank
        saved_sub_bank = load_sub_bank(n_clusters)
        if saved_sub_bank is not None:
            print(f"✅ Verification successful! Saved sub-bank shape: {saved_sub_bank.shape}")
        else:
            print("❌ Error: Sub-bank was not saved correctly!")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        return 1

    print("\n🎉 All done! You can now use the sub-bank for fast anomaly detection.")
    return 0


if __name__ == "__main__":
    exit(main())
