#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Defaults
export PYTHONPATH=${SCRIPT_DIR}/dlio_benchmark
CONFIG_PATH=${SCRIPT_DIR}/storage-conf

# TODO add DLRM when supported
WORKLOADS=("unet3d" "bert")

UNET3D_CONFIG_FILE=${CONFIG_PATH}/workload/unet3d.yaml
BERT_CONFIG_FILE=${CONFIG_PATH}/workload/bert.yaml
# Currently only "closed" category is supported
CATEGORIES=("closed" "open")
DEFAULT_CATEGORY="closed"
CLOSED_CATEGORY_PARAMS=(
	# dataset params
	"dataset.num_files_train" "dataset.num_subfolders_train" "dataset.data_folder"
	# reader params
	"reader.read_threads" "reader.computation_threads"
	# checkpoint params
	"checkpoint.checkpoint_folder"
	#storage params
	"storage.storage_type" "storage.storage_root")

OPEN_CATEGORY_PARAMS=(
	# all closed params
	"${CLOSED_CATEGORY_PARAMS[@]}"
	# framework params
	"framework"
	# dataset params
	"dataset.format" "dataset.num_samples_per_file"
	# reader params
	"reader.data_loader" "reader.transfer_size"
)
HYDRA_OUTPUT_CONFIG_DIR="configs"
EXTRA_PARAMS=(
	# benchmark do not rely on any profilers
	++workload.workflow.profiling=False
	++workload.profiling.profiler=none
	# config directory inside results used during runs
	++hydra.output_subdir=$HYDRA_OUTPUT_CONFIG_DIR
)
# currently only v100-32gb is supported
ACCELERATOR_TYPES=("v100-32gb")
DEFAULT_ACCELERATOR_TYPE="v100-32gb"
STEPS_PER_EPOCH=100
# host memory multiplier for dataset generation
HOST_MEMORY_MULTIPLIER=5

usage() {
	echo -e "Usage: $0 [datasize/datagen/run/configview/reportgen] [options]"
	echo -e "Script to launch the MLPerf Storage benchmark.\n"
}

datasize_usage() {
    echo -e "Usage: $0 datasize [options]"
    echo -e "Get minimum dataset size required for the benchmark run.\n"
    echo -e "\nOptions:"
    echo -e "  -h, --help\t\t\tPrint this message"
    echo -e "  -w, --workload\t\tWorkload dataset to be generated. Possible options are 'unet3d', 'bert' "
    echo -e "  -n, --num-accelerators\tSimulated number of accelerators per node of same accelerator type"
    echo -e "  -m, --host-memory-in-gb\tMemory available in the client where benchmark is run"
}

datagen_usage() {
	echo -e "Usage: $0 datagen [options]"
	echo -e "Generate benchmark dataset based on the specified options.\n"
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message"
	echo -e "  -c, --category\t\tBenchmark category to be submitted. Possible options are 'closed'(default)"
	echo -e "  -w, --workload\t\tWorkload dataset to be generated. Possible options are 'unet3d', 'bert' "
	echo -e "  -n, --num-parallel\t\tNumber of parallel jobs used to generate the dataset"
	echo -e "  -r, --results-dir\t\tLocation to the results directory. Default is ./results/workload.model/DATE-TIME"
	echo -e "  -p, --param\t\t\tDLIO param when set, will override the config file value"
}

run_usage() {
	echo -e "Usage: $0 run [options]"
	echo -e "Run benchmark on the generated dataset based on the specified options.\n"
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message"
	echo -e "  -c, --category\t\tBenchmark category to be submitted. Possible options are 'closed'(default)"
	echo -e "  -w, --workload\t\tWorkload to be run. Possible options are 'unet3d', 'bert' "
	echo -e "  -g, --accelerator-type\tSimulated accelerator type used for the benchmark. Possible options are 'v100-32gb'(default) "
	echo -e "  -n, --num-accelerators\tSimulated number of accelerators per node of same accelerator type"
	echo -e "  -r, --results-dir\t\tLocation to the results directory. Default is ./results/workload.model/DATE-TIME"
	echo -e "  -p, --param\t\t\tDLIO param when set, will override the config file value"
}

configview_usage() {
	echo -e "Usage: $0 configview [options]"
	echo -e "View the final config based on the specified options.\n"
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message"
	echo -e "  -w, --workload\t\tWorkload to be viewed. Possible options are 'unet3d', 'bert' "
	echo -e "  -p, --param\t\t\tDLIO param when set, will override the config file value"
}

reportgen_usage() {
	echo -e "Usage: $0 reportgen [options]"
	echo -e "Generate a report from the benchmark results.\n"
	echo -e "\nOptions:"
	echo -e "  -h, --help\t\t\tPrint this message"
	echo -e "  -r, --results-dir\t\tLocation to the results directory"
}


validate_in_list() {
	local typ=$1; shift
	local element=$1; shift
	list=("$@")
	if [[ ! " ${list[@]} " =~ " ${element} " ]]; then
		echo "argument ${element} for ${typ} is invalid. It has be one of (${list[*]})"
		exit 1
	fi
}

validate_category() {
	validate_in_list "category" $1 "${CATEGORIES[@]}" ;
}

validate_workload() {
	validate_in_list "workload" $1 "${WORKLOADS[@]}" ;
}

validate_accelerator_type() {
	validate_in_list "accelerator_type" $1 "${ACCELERATOR_TYPES[@]}" ;
}

validate_params() {
	local category=$1; shift
	local params=("$@")
	for param in "${params[@]}"
	do
		param_name=$(echo $param | cut -d '=' -f 1)
		if [[ " ${category} " =~ " open " ]]; then
			validate_in_list "params" $param_name "${OPEN_CATEGORY_PARAMS[@]}"
		elif [[ " ${category} " =~ " closed " ]]; then
			validate_in_list "params" $param_name "${CLOSED_CATEGORY_PARAMS[@]}"
		fi
		param_value=$(echo $param | cut -d '=' -f 2)
		validate_non_empty $param_name $param_value
	done
}



validate_non_empty() {
	local name=$1
	local value=$2
	if [[ -z "$value" ]]; then
		echo "${name} should not be empty"
		exit 1
	fi
}

add_prefix_params() {
	local input_array=( "$@" )
	local prefixed_array=()
	# prefix is fixed as per directory structure
	prefix="++workload."
	for element in "${input_array[@]}"; do
		prefixed_array+=("$prefix$element")
	done

	echo "${prefixed_array[@]}"
}

get_key_value_from_file() {
    local workload=$1;shift
    local key=$1;shift
    if [[ " ${workload} " =~ " unet3d " ]]; then
      value=`grep "$key:" $UNET3D_CONFIG_FILE| tr -d ' ' | cut -d':' -f2`
    elif [[ " ${workload} " =~ " bert " ]]; then
      value=`grep "$key:" $BERT_CONFIG_FILE| tr -d ' ' | cut -d':' -f2`
    fi
    echo "$value"
}

datasize() {
    local workload=$1;shift
    local num_accelerators=$1;shift
    local host_memory_in_gb=$1; shift

    num_steps_per_epoch=$(get_key_value_from_file $workload "total_training_steps")
    batch_size=$(get_key_value_from_file $workload "batch_size")
    record_length=$(get_key_value_from_file $workload "record_length")
    num_samples_per_file=$(get_key_value_from_file $workload "num_samples_per_file")
    computation_time=$(get_key_value_from_file $workload "computation_time")
    epochs=$(get_key_value_from_file $workload "epochs")
    if [[ -z "$num_steps_per_epoch" ]]; then
      # if total_training_steps is not set in config file, set num_steps_per_epoch to constant
      num_steps_per_epoch=${STEPS_PER_EPOCH}
      steps_per_epoch_from_config_file=0
    else
      # if total_training_steps is set in config file, set num_steps_per_epoch to value from config gile
      steps_per_epoch_from_config_file=1
    fi
    if [[ -z "$batch_size" ]]; then
      echo "Invalid config file. Batch size should not empty"
      exit 1
    fi
    if [[ -z "$epochs" ]]; then
      epochs=1
    fi
    # calculate required minimum samples given number of steps per epoch
    min_samples_steps_per_epoch=$(echo "$num_steps_per_epoch * $batch_size * $num_accelerators" | bc)
    # calculate required minimum samples given host memory to eliminate client-side caching effects
    min_samples_host_memory=$(echo "$host_memory_in_gb * $HOST_MEMORY_MULTIPLIER * 1024 * 1024 * 1024 / $record_length" | bc)
    # ensure we meet both constraints: min_samples = max(min_samples_v1, min_samples_v2)
    min_samples=$(( $min_samples_steps_per_epoch  > $min_samples_host_memory ? $min_samples_steps_per_epoch : $min_samples_host_memory ))
    # calculate minimum files to generate
    min_total_files=$(echo "$min_samples / $num_samples_per_file" | bc)
    min_files_size=$(echo "$min_samples * $record_length / 1024 / 1024 / 1024" | bc)
    # approx total time of the benchmark
    if [ $steps_per_epoch_from_config_file -eq 1 ]; then
      # if total_training_steps is set in config file, max time will be calculated on configured total_training_steps
      min_samples=$min_samples_steps_per_epoch
    fi
    total_time_min=$(echo "$min_samples / $batch_size / $num_accelerators * $computation_time * $epochs / 60" | bc)
    echo "The benchmark will run for approx ${total_time_min} minutes(best case)"
    echo "Minimum ${min_total_files} files are required which will consume ${min_files_size} GB of storage"
    echo "----------------------------------------------"
    echo "Set --param dataset.num_files_train=${min_total_files} with ./benchmark.sh datagen/run commands"

}

datagen() {
	local category=$1; shift
	local workload=$1;shift
	local parallel=$1;shift
	local results_dir=$1; shift
	local params=("$@")
	validate_category $category
	validate_workload $workload
	if [[ ! -z "$params" ]]; then
		validate_params $category "${params[@]}"
	fi
	if [[ ! -z "$results_dir" ]]; then
		EXTRA_PARAMS=(
			"${EXTRA_PARAMS[@]}" ++hydra.run.dir=$results_dir
		)
	fi
	prefixed_array=$(add_prefix_params ${params[@]})
	mpirun -np $parallel python3 dlio_benchmark/src/dlio_benchmark.py --config-path=$CONFIG_PATH workload=$workload ++workload.workflow.generate_data=True ++workload.workflow.train=False ${prefixed_array[@]} ${EXTRA_PARAMS[@]}
}

run() {
	local category=$1;shift
	local workload=$1;shift
	local accelerator_type=$1;shift
	local num_accelerators=$1;shift
	local results_dir=$1; shift
	local params=("$@")
	validate_category $category
	validate_workload $workload
	validate_accelerator_type $accelerator_type
	if [[ ! -z "$params" ]]; then
		validate_params $category "${params[@]}"
	fi
	if [[ ! -z "$results_dir" ]]; then
		EXTRA_PARAMS=(
			"${EXTRA_PARAMS[@]}" ++hydra.run.dir=$results_dir
		)
	fi
	prefixed_array=$(add_prefix_params ${params[@]})
	mpirun -np $num_accelerators python3 dlio_benchmark/src/dlio_benchmark.py --config-path=$CONFIG_PATH workload=$workload ++workload.workflow.generate_data=False ++workload.workflow.train=True ${prefixed_array[@]} ${EXTRA_PARAMS[@]}
	#python report.py --result-dir $results_dir --config-path=$CONFIG_PATH
}

configview() {
	local workload=$1;shift
	local params=("$@")
	validate_workload $workload
	prefixed_array=$(add_prefix_params ${params[@]})
	python3 dlio_benchmark/src/dlio_benchmark.py --config-path=$CONFIG_PATH workload=$workload ${prefixed_array[@]} --cfg=job
}

postprocess() {
	local results_dir=$1
	python3 report.py --result-dir $results_dir
}

main() {

	local mode=$1; shift
	if [ "$mode" = "datasize" ]
	then
		while [ $# -gt 0 ]; do
			case "$1" in
				-h | --help ) datasize_usage; exit 0 ;;
				-w | --workload ) workload="$2"; shift 2 ;;
                -n | --num-accelerators ) num_accelerators="$2"; shift 2 ;;
				-m | --host-memory-in-gb ) host_memory_in_gb="$2"; shift 2 ;;
				* ) echo "Invalid option $1"; datasize_usage; exit 1 ;;
			esac
		done
		validate_non_empty "workload" $workload
        validate_non_empty "num-accelerators" $num_accelerators
        validate_non_empty "host_memory_in_gb" $host_memory_in_gb
        datasize $workload $num_accelerators $host_memory_in_gb
    elif [ "$mode" = "datagen" ]
	then
		params=()
		while [ $# -gt 0 ]; do
			case "$1" in
				-h | --help ) datagen_usage; exit 0 ;;
				-c | --category ) category="$2"; shift 2 ;;
				-w | --workload ) workload="$2"; shift 2 ;;
				-n | --num-parallel ) parallel="$2"; shift 2 ;;
				-r | --results-dir ) results_dir="$2"; shift 2 ;;
				-p | --param ) params+=("$2"); shift 2 ;;
				* ) echo "Invalid option $1"; datagen_usage; exit 1 ;;
			esac
		done
		category=${category:-$DEFAULT_CATEGORY}
		validate_non_empty "workload" $workload
		parallel=${parallel:-1}
		datagen $category $workload $parallel "$results_dir" "${params[@]}"
	elif [ "$mode" = "run" ]
	then
		params=()
		while [ $# -gt 0 ]; do
			case "$1" in
				-h | --help ) run_usage; exit 0 ;;
				-c | --category ) category="$2"; shift 2 ;;
				-w | --workload ) workload="$2"; shift 2 ;;
				-g | --accelerator-type ) accelerator_type="$2"; shift 2 ;;
				-n | --num-accelerators ) num_accelerators="$2"; shift 2 ;;
				-r | --results-dir ) results_dir="$2"; shift 2 ;;
				-p | --param ) params+=("$2"); shift 2 ;;
				* ) echo "Invalid option $1"; run_usage; exit 1 ;;
			esac
		done
		category=${category:-$DEFAULT_CATEGORY}
		accelerator_type=${accelerator_type:-$DEFAULT_ACCELERATOR_TYPE}
		validate_non_empty "workload" $workload
		validate_non_empty "num-accelerators" $num_accelerators
		run $category $workload $accelerator_type $num_accelerators "$results_dir" "${params[@]}"
	elif [ "$mode" = "configview" ]
	then
		params=()
		while [ $# -gt 0 ]; do
			case "$1" in
				-h | --help ) configview_usage; exit 0 ;;
				-w | --workload ) workload="$2"; shift 2 ;;
				-p | --param ) params+=("$2"); shift 2 ;;
				* ) echo "Invalid option $1"; configview_usage; exit 1 ;;
			esac
		done
		validate_non_empty "workload" $workload
		configview $workload "${params[@]}"
	elif [ "$mode" = "reportgen" ]
	then
		while [ $# -gt 0 ]; do
			case "$1" in
				-h | --help ) reportgen_usage; exit 0 ;;
				-r | --results-dir ) results_dir="$2"; shift 2 ;;
				* ) echo "Invalid option $1"; reportgen_usage; exit 1 ;;
			esac
		done
		validate_non_empty "results-dir" $results_dir
		postprocess $results_dir
	else
		usage; exit 1
	fi

}


main $@
