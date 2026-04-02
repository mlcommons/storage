#!/usr/bin/env python3
import argparse
import logging
import sys
import os
import time
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility

# Add the parent directory to sys.path to import config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdbbench.config_loader import load_config, merge_config_with_args
from vdbbench.compact_and_watch import monitor_progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Load vectors into Milvus database")
    
    # Connection parameters
    parser.add_argument("--host", type=str, default="localhost", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    
    # Collection parameters
    parser.add_argument("--collection-name", type=str, help="Name of the collection to create")
    parser.add_argument("--dimension", type=int, help="Vector dimension")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of shards for the collection")
    parser.add_argument("--vector-dtype", type=str, default="float", choices=["FLOAT_VECTOR"],
                        help="Vector data type. Only FLOAT_VECTOR is supported for now")
    parser.add_argument("--force", action="store_true", help="Force recreate collection if it exists")
    
    # Data generation parameters
    parser.add_argument("--num-vectors", type=int, help="Number of vectors to generate")
    parser.add_argument("--distribution", type=str, default="uniform", 
                        choices=["uniform", "normal"], help="Distribution for vector generation")
    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for insertion")
    parser.add_argument("--chunk-size", type=int, default=1000000, help="Number of vectors to generate in each chunk (for memory management)")

    # Index parameters
    parser.add_argument("--index-type", type=str, default="DISKANN", help="Index type")
    parser.add_argument("--metric-type", type=str, default="COSINE", help="Metric type for index")
    parser.add_argument("--max-degree", type=int, default=16, help="DiskANN MaxDegree parameter")
    parser.add_argument("--search-list-size", type=int, default=200, help="DiskANN SearchListSize parameter")
    parser.add_argument("--M", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW efConstruction parameter")
    parser.add_argument("--inline-pq", type=int, default=16, help="AISAQ inline_pq parameter, performance(max_degree) vs scale(0) mode")
    
    # Monitoring parameters
    parser.add_argument("--monitor-interval", type=int, default=5, help="Interval in seconds for monitoring index building")
    parser.add_argument("--compact", action="store_true", help="Perform compaction after loading")
    
    # Configuration file
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # What-if option to print args and exit
    parser.add_argument("--what-if", action="store_true", help="Print the arguments after processing and exit")
    
    # Debug option to set logging level to DEBUG
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Track which arguments were explicitly set vs using defaults
    args.is_default = {
        'host': args.host == "localhost",
        'port': args.port == "19530",
        'num_shards': args.num_shards == 1,
        'vector_dtype': args.vector_dtype == "float",
        'distribution': args.distribution == "uniform",
        'batch_size': args.batch_size == 10000,
        'chunk_size': args.chunk_size == 1000000,
        'index_type': args.index_type == "DISKANN",
        'metric_type': args.metric_type == "COSINE",
        'max_degree': args.max_degree == 16,
        'search_list_size': args.search_list_size == 200,
        'M': args.M == 16,
        'ef_construction': args.ef_construction == 200,
        'inline_pq': args.inline_pq == 16,
        'monitor_interval': args.monitor_interval == 5,
        'compact': not args.compact,  # Default is False
        'force': not args.force,  # Default is False
        'what_if': not args.what_if,  # Default is False
        'debug': not args.debug  # Default is False
    }
    
    # Set logging level to DEBUG if --debug is specified
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Load configuration from YAML if specified
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    
    # If what-if is specified, print the arguments and exit
    if args.what_if:
        logger.info("Running in what-if mode. Printing arguments and exiting.")
        print("\nConfiguration after processing arguments and config file:")
        print("=" * 60)
        for key, value in vars(args).items():
            if key != 'is_default':  # Skip the is_default dictionary
                source = "default" if args.is_default.get(key, False) else "specified"
                print(f"{key}: {value} ({source})")
        print("=" * 60)
        sys.exit(0)
    
    # Validate required parameters
    required_params = ['collection_name', 'dimension', 'num_vectors']
    missing_params = [param for param in required_params if getattr(args, param.replace('-', '_'), None) is None]
    
    if missing_params:
        parser.error(f"Missing required parameters: {', '.join(missing_params)}. "
                     f"Specify with command line arguments or in config file.")
    
    return args


def connect_to_milvus(host, port):
    """Connect to Milvus server"""
    try:
        logger.debug(f"Connecting to Milvus server at {host}:{port}")
        connections.connect(
            "default", 
            host=host, 
            port=port,
            max_receive_message_length=514_983_574,
            max_send_message_length=514_983_574
        )
        logger.info(f"Connected to Milvus server at {host}:{port}")
        return True

    except Exception as e:
        logger.error(f"Error connecting to Milvus server: {str(e)}")
        return False


def create_collection(collection_name, dim, num_shards, vector_dtype, force=False):
    """Create a new collection with the specified parameters"""
    try:
        # Check if collection exists
        if utility.has_collection(collection_name):
            if force:
                Collection(name=collection_name).drop()
                logger.info(f"Dropped existing collection: {collection_name}")
            else:
                logger.warning(f"Collection '{collection_name}' already exists. Use --force to drop and recreate it.")
                return None

        # Define vector data type
        vector_type = DataType.FLOAT_VECTOR

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=vector_type, dim=dim)
        ]
        schema = CollectionSchema(fields, description="Benchmark Collection")

        # Create collection
        collection = Collection(name=collection_name, schema=schema, num_shards=num_shards)
        logger.info(f"Created collection '{collection_name}' with {dim} dimensions and {num_shards} shards")

        return collection
    except Exception as e:
        logger.error(f"Failed to create collection: {str(e)}")
        return None


def generate_vectors(num_vectors, dim, distribution='uniform'):
    """Generate random vectors based on the specified distribution.

    Returns a numpy float32 array (accepted directly by pymilvus),
    avoiding the huge memory overhead of converting to Python lists.
    """
    if distribution == 'uniform':
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)
    elif distribution == 'normal':
        vectors = np.random.normal(0, 1, (num_vectors, dim)).astype(np.float32)
    elif distribution == 'zipfian':
        # Simplified zipfian-like distribution
        base = np.random.random((num_vectors, dim)).astype(np.float32)
        skew = np.random.zipf(1.5, (num_vectors, 1)).astype(np.float32)
        vectors = base * (skew / 10)
    else:
        vectors = np.random.random((num_vectors, dim)).astype(np.float32)

    # Normalize vectors in-place to avoid extra allocations
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    return vectors


def insert_data(collection, vectors, batch_size=10000):
    """Insert vectors into the collection in batches"""
    total_vectors = len(vectors)
    num_batches = (total_vectors + batch_size - 1) // batch_size

    start_time = time.time()
    total_inserted = 0

    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, total_vectors)
        batch_size_actual = batch_end - batch_start

        # Prepare batch data
        ids = list(range(batch_start, batch_end))
        batch_vectors = vectors[batch_start:batch_end]

        # Insert batch
        try:
            collection.insert([ids, batch_vectors])
            total_inserted += batch_size_actual

            # Log progress
            progress = total_inserted / total_vectors * 100
            elapsed = time.time() - start_time
            rate = total_inserted / elapsed if elapsed > 0 else 0

            logger.info(f"Inserted batch {i+1}/{num_batches}: {progress:.2f}% complete, "
                        f"rate: {rate:.2f} vectors/sec")

        except Exception as e:
            logger.error(f"Error inserting batch {i+1}: {str(e)}")

    return total_inserted, time.time() - start_time


def flush_collection(collection):
    # Flush the collection
    flush_start = time.time()
    collection.flush()
    flush_time = time.time() - flush_start
    logger.info(f"Flush completed in {flush_time:.2f} seconds")


def create_index(collection, index_params):
    """Create an index on the collection"""
    try:
        start_time = time.time()
        logger.info(f"Creating index with parameters: {index_params}")
        collection.create_index("vector", index_params)
        index_creation_time = time.time() - start_time
        logger.info(f"Index creation command completed in {index_creation_time:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Failed to create index: {str(e)}")
        return False


def main():
    args = parse_args()

    # Connect to Milvus
    if not connect_to_milvus(args.host, args.port):
        logger.error("Failed to connect to Milvus.")
        return 1

    logger.debug(f'Determining datatype for vector representation.')
    # Determine vector data type
    try:
        # Check if FLOAT16 is available in newer versions of pymilvus
        if hasattr(DataType, 'FLOAT16'):
            logger.debug(f'Using FLOAT16 data type for vector representation.")')
            vector_dtype = DataType.FLOAT16 if args.vector_dtype == 'float16' else DataType.FLOAT_VECTOR
        else:
            # Fall back to supported data types
            logger.warning("FLOAT16 data type not available in this version of pymilvus. Using FLOAT_VECTOR instead.")
            vector_dtype = DataType.FLOAT_VECTOR
    except Exception as e:
        logger.warning(f"Error determining vector data type: {str(e)}. Using FLOAT_VECTOR as default.")
        vector_dtype = DataType.FLOAT_VECTOR

    # Create collection
    collection = create_collection(
        collection_name=args.collection_name,
        dim=args.dimension,
        num_shards=args.num_shards,
        vector_dtype=vector_dtype,
        force=args.force
    )

    if collection is None:
        return 1

    # Create index with updated parameters
    index_params = {
        "index_type": args.index_type,
        "metric_type": args.metric_type,
        "params": {}
    }

    # Update only the parameters based on index_type
    if args.index_type == "HNSW":
        index_params["params"] = {
            "M": args.M,
            "efConstruction": args.ef_construction
        }
    elif args.index_type == "DISKANN":
        index_params["params"] = {
            "MaxDegree": args.max_degree,
            "SearchListSize": args.search_list_size
        }
    elif args.index_type == "AISAQ":
        index_params["params"] = {
            "inline_pq": args.inline_pq,
            "max_degree": args.max_degree,
            "search_list_size": args.search_list_size
        }
    else:
        raise ValueError(f"Unsupported index_type: {args.index_type}")

    logger.debug(f'Creating index. This should be immediate on an empty collection')
    if not create_index(collection, index_params):
        return 1

    # Generate vectors
    logger.info(
        f"Generating {args.num_vectors} vectors with {args.dimension} dimensions using {args.distribution} distribution")
    start_gen_time = time.time()
    
    # Split vector generation into chunks if num_vectors is large
    if args.num_vectors > args.chunk_size:
        logger.info(f"Large vector count detected. Generating in chunks of {args.chunk_size:,} vectors")
        vectors = []
        remaining = args.num_vectors
        chunks_processed = 0
        
        while remaining > 0:
            chunk_size = min(args.chunk_size, remaining)
            logger.info(f"Generating chunk {chunks_processed+1}: {chunk_size:,} vectors")
            chunk_start = time.time()
            chunk_vectors = generate_vectors(chunk_size, args.dimension, args.distribution)
            chunk_time = time.time() - chunk_start

            logger.info(f"Generated chunk {chunks_processed} ({chunk_size:,} vectors) in {chunk_time:.2f} seconds. "
                        f"Progress: {(args.num_vectors - remaining):,}/{args.num_vectors:,} vectors "
                        f"({(args.num_vectors - remaining) / args.num_vectors * 100:.1f}%)")

            # Insert data
            logger.info(f"Inserting {args.num_vectors} vectors into collection '{args.collection_name}'")
            total_inserted, insert_time = insert_data(collection, chunk_vectors, args.batch_size)
            logger.info(f"Inserted {total_inserted} vectors in {insert_time:.2f} seconds")

            remaining -= chunk_size
            chunks_processed += 1
    else:
        # For smaller vector counts, generate all at once
        vectors = generate_vectors(args.num_vectors, args.dimension, args.distribution)
        # Insert data
        logger.info(f"Inserting {args.num_vectors} vectors into collection '{args.collection_name}'")
        total_inserted, insert_time = insert_data(collection, vectors, args.batch_size)
        logger.info(f"Inserted {total_inserted} vectors in {insert_time:.2f} seconds")

    gen_time = time.time() - start_gen_time
    logger.info(f"Generated all {args.num_vectors:,} vectors in {gen_time:.2f} seconds")

    flush_collection(collection)

    # Monitor index building
    logger.info(f"Starting to monitor index building progress (checking every {args.monitor_interval} seconds)")
    monitor_progress(args.collection_name, args.monitor_interval, zero_threshold=10)

    if args.compact:
        logger.info(f"Compacting collection '{args.collection_name}'")
        collection.compact()
        monitor_progress(args.collection_name, args.monitor_interval, zero_threshold=30)
        logger.info(f"Collection '{args.collection_name}' compacted successfully.")

    # Summary
    logger.info("Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
