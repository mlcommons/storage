#!/usr/bin/env python3
import argparse
import logging
import os
import sys
import time

from datetime import datetime, timedelta
from pymilvus import connections, Collection, utility

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the parent directory to sys.path to import config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdbbench.config_loader import load_config, merge_config_with_args

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor Milvus collection compaction process")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Milvus server host")
    parser.add_argument("--port", type=str, default="19530", help="Milvus server port")
    parser.add_argument("--collection", type=str, required=False, help="Collection name to compact and monitor")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--compact", action="store_true", help="Perform compaction before monitoring")
    parser.add_argument("--zero-threshold", type=int, default=90,
                        help="Time in seconds to wait with zero pending rows before considering complete")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    args = parser.parse_args()

    # Track which arguments were explicitly set vs using defaults
    args.is_default = {
        'host': args.host == "127.0.0.1",
        'port': args.port == "19530",
        'interval': args.interval == 5,
        'zero_threshold': args.zero_threshold == 90,
        'compact': not args.compact  # Default is False
    }

    # Load configuration from YAML if specified
    config = {}
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)

    # Validate required parameters
    if not args.collection:
        parser.error("Collection name is required. Specify with --collection or in config file.")

    return args


def connect_to_milvus(host, port):
    """Connect to Milvus server"""
    try:
        connections.connect(
            "default",
            host=host,
            port=port,
            max_receive_message_length=514_983_574,
            max_send_message_length=514_983_574
        )
        logging.info(f"Connected to Milvus server at {host}:{port}")
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {str(e)}")
        return False

def perform_compaction(collection_name):
    """Perform compaction on the collection"""
    try:
        collection = Collection(name=collection_name)
        logging.info(f"Starting compaction on collection: {collection_name}")
        compaction_start = time.time()
        collection.compact()
        compaction_time = time.time() - compaction_start
        logging.info(f"Compaction command completed in {compaction_time:.2f} seconds")
        return True
    except Exception as e:
        logging.error(f"Failed to perform compaction: {str(e)}")
        return False

def monitor_progress(collection_name, interval=60, zero_threshold=300):
    """Monitor the progress of index building/compaction"""
    start_time = time.time()
    prev_check_time = start_time
    
    try:
        # Get initial progress
        prev_progress = utility.index_building_progress(collection_name=collection_name)
        initial_indexed_rows = prev_progress.get("indexed_rows", 0)
        initial_pending_rows = prev_progress.get("pending_index_rows", 0)
        total_rows = prev_progress.get("total_rows", 0)

        logging.info(f"Starting to monitor progress for collection: {collection_name}")
        logging.info(f"Initial state: {initial_indexed_rows:,} of {total_rows:,} rows indexed")
        logging.info(f"Initial pending rows: {initial_pending_rows:,}")

        # Track the phases
        indexing_phase_complete = initial_indexed_rows >= total_rows
        pending_phase_complete = False
        
        # Track time with zero pending rows
        pending_zero_start_time = None

        while True:
            time.sleep(interval)  # Check at specified interval
            current_time = time.time()
            elapsed_time = current_time - start_time
            time_since_last_check = current_time - prev_check_time
            
            try:
                progress = utility.index_building_progress(collection_name=collection_name)
                
                # Calculate progress metrics
                indexed_rows = progress.get("indexed_rows", 0)
                total_rows = progress.get("total_rows", total_rows)  # Use previous if not available
                pending_rows = progress.get("pending_index_rows", 0)

                # Quick exit:
                if pending_rows == 0 and indexed_rows == total_rows:
                    # Ensure the pending counter has started
                    if not pending_zero_start_time:
                        pending_zero_start_time = current_time
                        logging.info("No pending rows detected. Assuming indexing phase is complete.")
                        indexing_phase_complete = True
                
                # Calculate both overall and recent indexing rates
                total_rows_indexed_since_start = indexed_rows - initial_indexed_rows
                rows_since_last_check = indexed_rows - prev_progress.get("indexed_rows", indexed_rows)
                
                # Calculate pending rows reduction
                pending_rows_reduction = prev_progress.get("pending_index_rows", pending_rows) - pending_rows
                pending_reduction_rate = pending_rows_reduction / time_since_last_check if time_since_last_check > 0 else 0

                # Calculate overall rate (based on total time since monitoring began)
                if elapsed_time > 0:
                    # Calculate percent done regardless of whether new rows were indexed
                    percent_done = indexed_rows / total_rows * 100 if total_rows > 0 else 100
                    
                    if total_rows_indexed_since_start > 0:
                        # Normal case: some rows have been indexed since we started monitoring
                        overall_indexing_rate = total_rows_indexed_since_start / elapsed_time  # rows per second
                        remaining_rows = total_rows - indexed_rows
                        estimated_seconds_remaining = remaining_rows / overall_indexing_rate if overall_indexing_rate > 0 else float('inf')
                        
                        # Alternative estimate based on pending rows
                        pending_estimate = pending_rows / pending_reduction_rate if pending_reduction_rate > 0 and pending_rows > 0 else float('inf')
                        
                        # Calculate recent rate (for comparison)
                        recent_indexing_rate = rows_since_last_check / time_since_last_check if time_since_last_check > 0 else 0
                        
                        # Format the estimated time remaining
                        eta = datetime.now() + timedelta(seconds=estimated_seconds_remaining)
                        eta_str = eta.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Format the pending-based estimate
                        pending_eta = datetime.now() + timedelta(seconds=pending_estimate) if pending_estimate != float('inf') else "Unknown"
                        if isinstance(pending_eta, datetime):
                            pending_eta_str = pending_eta.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            pending_eta_str = str(pending_eta)
                        
                        # Log progress with estimates
                        if not indexing_phase_complete:
                            # Still in initial indexing phase
                            logging.info(
                                f"Phase 1 - Building index: {percent_done:.2f}% complete... "
                                f"({indexed_rows:,}/{total_rows:,} rows) | "
                                f"Pending rows: {pending_rows:,} | "
                                f"Overall rate: {overall_indexing_rate:.2f} rows/sec | "
                                f"Recent rate: {recent_indexing_rate:.2f} rows/sec | "
                                f"ETA: {eta_str} | "
                                f"Est. remaining: {timedelta(seconds=int(estimated_seconds_remaining))}"
                            )
                        else:
                            # In pending rows processing phase
                            if pending_rows > 0:
                                # Reset the zero pending timer if we see pending rows
                                pending_zero_start_time = None
                                
                                logging.info(
                                    f"Phase 2 - Processing pending rows: {pending_rows:,} remaining | "
                                    f"Reduction rate: {pending_reduction_rate:.2f} rows/sec | "
                                    f"ETA: {pending_eta_str} | "
                                    f"Est. remaining: {timedelta(seconds=int(pending_estimate)) if pending_estimate != float('inf') else 'Unknown'}"
                                )
                            else:
                                # Handle zero pending rows case (same as below)
                                if pending_zero_start_time is None:
                                    pending_zero_start_time = current_time
                                    logging.info(f"No pending rows detected. Starting {zero_threshold//60}-minute confirmation timer.")
                                else:
                                    zero_pending_time = current_time - pending_zero_start_time
                                    logging.info(f"No pending rows for {zero_pending_time:.1f} seconds (waiting for {zero_threshold} seconds to confirm)")
                                    
                                    if zero_pending_time >= zero_threshold:
                                        logging.info(f"No pending rows detected for {zero_threshold//60} minutes. Process is considered complete.")
                                        pending_phase_complete = True
                    else:
                        # Special case: all rows were already indexed when we started monitoring
                        logging.info(
                            f"Progress: {percent_done:.2f}% complete... "
                            f"({indexed_rows:,}/{total_rows:,} rows) | "
                            f"Pending rows: {pending_rows:,}"
                        )
                        
                        # If all rows are indexed and there are no pending rows, we might be done
                        if indexed_rows >= total_rows and pending_rows == 0:
                            if not indexing_phase_complete:
                                indexing_phase_complete = True
                                logging.info(f"Initial indexing phase complete! All {indexed_rows:,} rows have been indexed.")
                            
                            # Handle zero pending rows case
                            if pending_zero_start_time is None:
                                pending_zero_start_time = current_time
                                logging.info(f"No pending rows detected. Starting {zero_threshold}-second confirmation timer.")
                            else:
                                zero_pending_time = current_time - pending_zero_start_time
                                logging.info(f"No pending rows for {zero_pending_time:.1f} seconds (waiting for {zero_threshold} seconds to confirm)")
                                
                                if zero_pending_time >= zero_threshold:
                                    logging.info(f"No pending rows detected for {zero_threshold} seconds. Process is considered complete.")
                                    pending_phase_complete = True
                else:
                    # If no time has elapsed (first iteration)
                    percent_done = indexed_rows / total_rows * 100 if total_rows > 0 else 0
                    logging.info(
                        f"Progress: {percent_done:.2f}% complete... "
                        f"({indexed_rows:,}/{total_rows:,} rows) | "
                        f"Pending rows: {pending_rows:,} | "
                        f"Initial measurement, no progress data yet"
                    )
                
                # Check if pending phase is complete
                if not pending_phase_complete and pending_rows == 0:
                    # If we've already waited long enough with zero pending rows
                    if pending_zero_start_time is not None and (current_time - pending_zero_start_time) >= zero_threshold:
                        pending_phase_complete = True
                        logging.info(f"Pending rows processing complete! All pending rows have been processed.")
                
                # Check if both phases are complete
                if (indexed_rows >= total_rows or indexing_phase_complete) and pending_phase_complete:
                    total_time = time.time() - start_time
                    logging.info(f"Process fully complete! Total time: {timedelta(seconds=int(total_time))}")
                    break
                    
                # Update for next iteration
                prev_progress = progress
                prev_check_time = current_time
                
            except Exception as e:
                logging.error(f"Error checking progress: {str(e)}")
                time.sleep(5)  # Short delay before retrying
                
    except Exception as e:
        logging.error(f"Error in monitor_progress: {str(e)}")
        return False
    
    return True

def main():
    args = parse_args()
    
    # Connect to Milvus
    if not connect_to_milvus(args.host, args.port):
        return 1
    
    # Perform compaction if requested
    if args.compact:
        if not perform_compaction(args.collection):
            return 1
    
    # Monitor progress
    logging.info(f"Starting to monitor progress (checking every {args.interval} seconds)")
    if not monitor_progress(args.collection, args.interval, args.zero_threshold):
        return 1
    
    logging.info("Monitoring completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
