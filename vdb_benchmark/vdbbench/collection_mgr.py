#!/usr/bin/env python3
"""
milvus_interactive_col_mgr.py
------------------------------------
* **Back to list** — press **b** inside the operations menu to return to the
  collection picker without quitting the program.
* **Enhanced index support** — displays parameters for HNSW, DiskANN, and AISAQ
* **Dynamic vector field detection** — automatically finds vector field
* **Improved error handling** — better exception handling throughout

Requires: pymilvus, tabulate, numpy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from pymilvus import Collection, connections, utility, DataType
from tabulate import tabulate

METRICS_PORT = 9091       # override with --metrics-port if needed

###############################################################################
# Conn helpers
###############################################################################

def connect(host: str, port: int) -> bool:
    """Connect to Milvus server with error handling"""
    try:
        if not connections.has_connection("default"):
            connections.connect("default", host=host, port=port)
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

###############################################################################
# Vector field detection (Correct / Enum-safe)
###############################################################################

from pymilvus import DataType
from typing import Optional, Tuple


def _dtype_to_str(dt) -> str:
    """
    Convert DataType enum or int to string safely across pymilvus versions.
    """
    if hasattr(dt, "name"):
        return dt.name  # modern enum case
    try:
        return DataType(dt).name  # fallback for int-like
    except Exception:
        return str(dt)


def _is_vector_dtype(dt) -> bool:
    """
    Check if dtype is any supported vector type (robust across versions).
    """
    vector_types = {
        DataType.FLOAT_VECTOR,
        DataType.BINARY_VECTOR,
    }

    # Optional types depending on Milvus version
    if hasattr(DataType, "FLOAT16_VECTOR"):
        vector_types.add(DataType.FLOAT16_VECTOR)
    if hasattr(DataType, "BFLOAT16_VECTOR"):
        vector_types.add(DataType.BFLOAT16_VECTOR)

    return dt in vector_types


def _get_vector_field_info(col: Collection) -> Tuple[Optional[str], Optional[str]]:
    """
    Dynamically find the vector field and return:
        (field_name, dtype_string)
    """
    try:
        for field in col.schema.fields:
            if _is_vector_dtype(field.dtype):
                return field.name, _dtype_to_str(field.dtype)
        return None, None
    except Exception:
        return None, None


def _get_vector_field(col: Collection) -> Optional[str]:
    """
    Get just the vector field name.
    """
    field_name, _ = _get_vector_field_info(col)
    return field_name


###############################################################################
# Status helpers
###############################################################################

def _is_loaded(col: Collection) -> bool:
    """Check if a collection is loaded"""
    try:
        if hasattr(col, "get_load_state"):
            return col.get_load_state().name == "Loaded"
        if hasattr(col, "load_state"):
            return col.load_state.name == "Loaded"
        # Fallback: try to get the load state via utility
        state = utility.load_state(col.name)
        return state.name == "Loaded"
    except Exception:
        return False


def _get_load_status(col: Collection) -> str:
    """Get load status as string"""
    return "✓ Loaded" if _is_loaded(col) else "Released"

###############################################################################
# Index parameters
###############################################################################

def _index_params(col: Collection) -> Tuple[str, str, str, str]:
    """Extract index parameters supporting multiple index types"""
    if not col.indexes:
        return "—", "—", "—", "—"
    
    try:
        p = col.indexes[0].params
        idx_type = p.get("index_type", "?")
        metric = p.get("metric_type", "?")
        
        params = p.get("params", {})
        # Support multiple index types
        if idx_type == "HNSW":
            param1 = params.get("M", "—")
            param2 = params.get("efConstruction", "—")
        elif idx_type == "DISKANN":
            # Support both PascalCase (old) and snake_case (new) parameter names
            param1 = params.get("max_degree", params.get("MaxDegree", "—"))
            param2 = params.get("search_list_size", params.get("SearchListSize", "—"))
        elif idx_type == "AISAQ":
            param1 = params.get("inline_pq", "—")
            param2 = params.get("max_degree", "—")
        elif idx_type == "IVF_FLAT" or idx_type == "IVF_SQ8" or idx_type == "IVF_PQ":
            param1 = params.get("nlist", "—")
            param2 = params.get("m", "—") if "m" in params else "—"
        else:
            param1 = param2 = "—"
        
        return idx_type, metric, str(param1), str(param2)
    except Exception as e:
        return "?", "?", "?", "?"

###############################################################################
# Inventory
###############################################################################

def inventory(host: str, metrics_port: int) -> List[dict]:
    """Get inventory of all collections with their details"""
    rows = []
    
    try:
        collection_names = utility.list_collections()
    except Exception as e:
        print(f"❌ Failed to list collections: {e}")
        return []
    
    for name in collection_names:
        try:
            col = Collection(name)
            idx_type, metric, param1, param2 = _index_params(col)
            
            # Get vector field info
            vector_field, data_type = _get_vector_field_info(col)
            dim = "—"
            if vector_field:
                for f in col.schema.fields:
                    if f.name == vector_field:
                        dim = f.params.get("dim", "—")
                        break
            
            # Get load status
            load_status = _get_load_status(col)
            
            rows.append(
                dict(
                    name=name,
                    entities=f"{col.num_entities:,}",
                    dim=dim,
                    data_type=data_type or "—",
                    idx_type=idx_type,
                    metric=metric,
                    connectivity=param1,
                    build_quality=param2,
                    load_status=load_status,
                )
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to get info for collection '{name}': {e}")
            continue
    
    return rows

###############################################################################
# Picker
###############################################################################

def pick_collection(host: str, metrics_port: int) -> Collection | None:
    """Interactive collection picker"""
    inv = inventory(host, metrics_port)
    if not inv:
        print("❌ No collections found.")
        return None

    headers = [
        "Idx",
        "Collection",
        "Entities",
        "Dim",
        "DataType",
        "IdxType",
        "Metric",
        "Connectivity",
        "IdxBuild",
        "Status",
    ]
    rows = [
        [
            i,
            r["name"],
            r["entities"],
            r["dim"],
            r["data_type"],
            r["idx_type"],
            r["metric"],
            r["connectivity"],
            r["build_quality"],
            r["load_status"],
        ]
        for i, r in enumerate(inv)
    ]
    print(tabulate(rows, headers=headers, tablefmt="github"))

    try:
        idx = int(input("\nSelect collection index › ").strip())
        if idx < 0 or idx >= len(inv):
            print("❌ Invalid index.")
            return None
        return Collection(inv[idx]["name"])
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None
    except Exception as e:
        print(f"❌ Error selecting collection: {e}")
        return None

###############################################################################
# Operations
###############################################################################

def validate_collection(col: Collection) -> bool:
    """Validate that a collection still exists and is accessible"""
    try:
        _ = col.num_entities
        return True
    except Exception:
        print("❌ Collection no longer exists or is inaccessible.")
        return False


def _loaded(col: Collection) -> bool:
    """Check if a collection is loaded"""
    return _is_loaded(col)


def op_load(col: Collection):
    """Load a collection into memory"""
    if not validate_collection(col):
        return
    
    try:
        if _loaded(col):
            print("✔ Already loaded.")
        else:
            col.load()
            print("[+] Loaded.")
    except Exception as e:
        print(f"❌ Load failed: {e}")


def op_release(col: Collection):
    """Release a collection from memory"""
    if not validate_collection(col):
        return
    
    try:
        if not _loaded(col):
            print("✔ Already released.")
        else:
            col.release()
            print("[−] Released.")
    except Exception as e:
        print(f"❌ Release failed: {e}")


def op_warm(col: Collection, n=5):
    """Warm up a collection with dummy queries"""
    if not validate_collection(col):
        return
    
    try:
        op_load(col)
        
        # Find vector field dynamically
        vector_field = _get_vector_field(col)
        if not vector_field:
            print("❌ No vector field found in collection.")
            return
        
        # Get dimension
        dim = None
        for f in col.schema.fields:
            if f.name == vector_field:
                dim = f.params.get("dim")
                break
        
        if not dim:
            print("❌ Could not determine vector dimension.")
            return
        
        # Get collection's metric type from index
        metric_type = "L2"
        search_params = {"ef": 16}
        
        if col.indexes:
            idx_params = col.indexes[0].params
            metric_type = idx_params.get("metric_type", "L2")
            idx_type = idx_params.get("index_type", "")
            
            # Adjust search params based on index type
            if idx_type == "HNSW":
                search_params = {"ef": 64}
            elif idx_type == "DISKANN":
                search_params = {"search_list": 100}
            elif idx_type.startswith("IVF"):
                search_params = {"nprobe": 10}
        
        # Generate and execute dummy queries
        dummy = np.random.random((n, dim)).astype(np.float32).tolist()
        _ = col.search(
            dummy, 
            vector_field, 
            {"metric_type": metric_type, "params": search_params}, 
            limit=1
        )
        print(f"[✓] Warmed ({n} dummy queries with {metric_type} metric).")
    except Exception as e:
        print(f"❌ Warm failed: {e}")


def op_delete(col: Collection):
    """Delete (drop) a collection"""
    if not validate_collection(col):
        return
    
    try:
        confirm = input(f"⚠ Really DROP collection '{col.name}'? (yes/[no]) › ").strip().lower()
        if confirm == "yes":
            col.drop()
            print("[×] Collection dropped.")
        else:
            print("✓ Aborted; collection kept.")
    except Exception as e:
        print(f"❌ Delete failed: {e}")


def op_compact(col: Collection):
    """Compact a collection"""
    if not validate_collection(col):
        return
    
    try:
        print(f"⏳ Starting compaction on '{col.name}'...")
        col.compact()
        print(f"[✓] Compaction initiated. Use monitoring tools to track progress.")
    except Exception as e:
        print(f"❌ Compact failed: {e}")


def op_info(col: Collection):
    """Display detailed information about a collection"""
    if not validate_collection(col):
        return
    
    try:
        print(f"\n{'='*70}")
        print(f"Collection: {col.name}")
        print(f"{'='*70}")
        print(f"Entities: {col.num_entities:,}")
        print(f"Loaded: {'Yes' if _loaded(col) else 'No'}")
        
        # Schema info
        print(f"\nSchema:")
        for field in col.schema.fields:
            field_type = field.dtype
            extra = f" (dim={field.params.get('dim')})" if field.params.get('dim') else ""
            primary = " [PRIMARY]" if field.is_primary else ""
            print(f"  - {field.name}: {field_type}{extra}{primary}")
        
        # Index info
        if col.indexes:
            print(f"\nIndex:")
            for idx in col.indexes:
                idx_type = idx.params.get('index_type', 'UNKNOWN')
                metric_type = idx.params.get('metric_type', 'UNKNOWN')
                params = idx.params.get('params', {})
                
                print(f"  Field: {idx.field_name}")
                print(f"  Type: {idx_type}")
                print(f"  Metric: {metric_type}")
                
                # Display build-time parameters
                print(f"  Build Parameters:")
                if idx_type == "HNSW":
                    print(f"    - M: {params.get('M', '—')}")
                    print(f"    - efConstruction: {params.get('efConstruction', '—')}")
                    
                elif idx_type == "DISKANN":
                    # Support both PascalCase (old) and snake_case (new) parameter names
                    max_deg = params.get('max_degree', params.get('MaxDegree', '—'))
                    search_list = params.get('search_list_size', params.get('SearchListSize', '—'))
                    print(f"    - max_degree: {max_deg}")
                    print(f"    - search_list_size: {search_list}")
                    
                elif idx_type == "AISAQ":
                    print(f"    - inline_pq: {params.get('inline_pq', '—')}")
                    print(f"    - max_degree: {params.get('max_degree', '—')}")
                    print(f"    - search_list_size: {params.get('search_list_size', '—')}")
                    
                elif idx_type.startswith("IVF"):
                    print(f"    - nlist: {params.get('nlist', '—')}")
                    if 'm' in params:
                        print(f"    - m: {params.get('m', '—')}")
                    if 'nbits' in params:
                        print(f"    - nbits: {params.get('nbits', '—')}")
                        
                else:
                    # Generic display for unknown index types
                    for key, value in params.items():
                        print(f"    - {key}: {value}")
                
        else:
            print(f"\nIndex: None")
        
        # Partitions
        print(f"\nPartitions: {len(col.partitions)}")
        for partition in col.partitions:
            print(f"  - {partition.name}")
        
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"❌ Info failed: {e}")

###############################################################################
# Main CLI loop
###############################################################################

def main():
    ap = argparse.ArgumentParser(description="Interactive Milvus collection manager")
    ap.add_argument("--host", default="localhost", help="Milvus host (default: localhost)")
    ap.add_argument("--port", type=int, default=19530, help="Milvus port (default: 19530)")
    ap.add_argument("--metrics-port", type=int, default=METRICS_PORT, 
                    help=f"Prometheus metrics port (default: {METRICS_PORT})")
    args = ap.parse_args()

    if not connect(args.host, args.port):
        sys.exit(1)

    while True:
        col = pick_collection(args.host, args.metrics_port)
        if col is None:
            sys.exit(1)

        menu = {
            "l": ("load", op_load),
            "r": ("release", op_release),
            "w": ("warm", op_warm),
            "c": ("compact", op_compact),
            "i": ("info", op_info),
            "d": ("delete", op_delete),
            "b": ("back", lambda c: None),
            "q": ("quit", lambda c: None),
        }

        while True:
            print("\nOperations: " + ", ".join([f"{k}={v[0]}" for k, v in menu.items()]))
            choice = input("Enter choice › ").strip().lower()
            
            if choice not in menu:
                print("❌ Unknown option.")
                continue
            
            if choice == "q":
                print("👋 Bye.")
                sys.exit(0)
            
            if choice == "b":
                break  # back to collection list
            
            menu[choice][1](col)


if __name__ == "__main__":
    main()