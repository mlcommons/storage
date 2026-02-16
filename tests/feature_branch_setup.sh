#!/bin/bash
# Setup feature branches for separate PRs

echo "Creating feature branches for clean PRs..."

# Feature 1: Multi-library storage (already on TF_ObjectStorage)
git checkout TF_ObjectStorage
git branch feature/multi-library-storage || echo "Branch already exists"

# Feature 2: Checkpoint optimization (from streaming-checkpoint-poc)
git checkout streaming-checkpoint-poc  
git branch feature/checkpoint-dgen-optimization || echo "Branch already exists"

# Return to working branch
git checkout TF_ObjectStorage

echo ""
echo "✅ Feature branches created:"
echo "   - feature/multi-library-storage (from TF_ObjectStorage)"
echo "   - feature/checkpoint-dgen-optimization (from streaming-checkpoint-poc)"
echo ""
echo "Next steps:"
echo "  1. Review/test feature/multi-library-storage"
echo "  2. Review/test feature/checkpoint-dgen-optimization"  
echo "  3. Push both branches and create PRs"
echo "  4. Merge both into TF_ObjectStorage for integration testing"
