#!/usr/bin/env python3
"""
Milvus Collection Information Script

This script connects to a Milvus instance and lists all collections with detailed information
including the number of vectors in each collection and index information.
"""

import sys
import os
import argparse
import logging
from tabulate import tabulate
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to import config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pymilvus import connections, utility, Collection
except ImportError:
    logger.error("Error: pymilvus package not found. Please install it with 'pip install pymilvus'")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    logger.error("Error: tabulate package not found. Please install it with 'pip install tabulate'")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="List Milvus collections with detailed information")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    parser.add_argument("--format", type=str, choices=["table", "json"], default="table", 
                        help="Output format (table or json)")
    return parser.parse_args()


def connect_to_milvus(host, port):
    """Connect to Milvus server"""
    try:
        connections.connect(
            alias="default", 
            host=host, 
            port=port
        )
        logger.info(f"Connected to Milvus server at {host}:{port}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus server: {str(e)}")
        return False


def get_collection_info(collection_name, release=True):
    """Get detailed information about a collection"""
    try:
        collection = Collection(collection_name)
        # collection.load()
        
        # Get basic collection info - using num_entities instead of get_statistics
        row_count = collection.num_entities
        # row_count = get_collection_info(collection_name)["row_count"]

        # Get schema information
        schema = collection.schema
        dimension = None
        for field in schema.fields:
            if field.dtype in [100, 101]:  # FLOAT_VECTOR or BINARY_VECTOR
                dimension = field.params.get("dim")
                break
        
        # Get index information
        index_info = []
        if collection.has_index():
            index = collection.index()
            index_info.append({
                "field_name": index.field_name,
                "index_type": index.params.get("index_type"),
                "metric_type": index.params.get("metric_type"),
                "params": index.params.get("params", {})
            })

        # Get partition information
        partitions = collection.partitions
        partition_info = [{"name": p.name, "description": p.description} for p in partitions]
        
        return {
            "name": collection_name,
            "row_count": row_count,
            "dimension": dimension,
            "schema": str(schema),
            "index_info": index_info,
            "partitions": partition_info
        }
    except Exception as e:
        logger.error(f"Error getting info for collection {collection_name}: {str(e)}")
        return {
            "name": collection_name,
            "error": str(e)
        }
    finally:
        # Release collection
        if release:
            try:
                collection.release()
            except:
                pass


def main():
    """Main function"""
    args = parse_args()
    
    # Connect to Milvus
    if not connect_to_milvus(args.host, args.port):
        return 1
    
    # List all collections
    try:
        collection_names = utility.list_collections()
        logger.info(f"Found {len(collection_names)} collections")
        
        if not collection_names:
            logger.info("No collections found in the Milvus instance")
            return 0
        
        # Get detailed information for each collection
        collections_info = []
        for name in collection_names:
            logger.info(f"Getting information for collection: {name}")
            info = get_collection_info(name)
            collections_info.append(info)
        
        # Display information based on format
        if args.format == "json":
            import json
            print(json.dumps(collections_info, indent=2))
        else:
            # Table format
            table_data = []
            for info in collections_info:
                index_types = ", ".join([idx.get("index_type", "N/A") for idx in info.get("index_info", [])])
                metric_types = ", ".join([idx.get("metric_type", "N/A") for idx in info.get("index_info", [])])
                
                row = [
                    info["name"],
                    info.get("row_count", "N/A"),
                    info.get("dimension", "N/A"),
                    index_types,
                    metric_types,
                    len(info.get("partitions", []))
                ]
                table_data.append(row)
            
            headers = ["Collection Name", "Vector Count", "Dimension", "Index Types", "Metric Types", "Partitions"]
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        return 0
    
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        return 1
    finally:
        # Disconnect from Milvus
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus server")
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())