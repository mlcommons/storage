import concurrent.futures
import pprint
import psutil
import os
import yaml


def read_config_from_file(relative_path):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def update_nested_dict(original_dict, update_dict):
    updated_dict = {}
    for key, value in original_dict.items():
        if key in update_dict:
            if isinstance(value, dict) and isinstance(update_dict[key], dict):
                updated_dict[key] = update_nested_dict(value, update_dict[key])
            else:
                updated_dict[key] = update_dict[key]
        else:
            updated_dict[key] = value
    for key, value in update_dict.items():
        if key not in original_dict:
            updated_dict[key] = value
    return updated_dict


def create_nested_dict(flat_dict, parent_dict=None, separator='.'):
    if parent_dict is None:
        parent_dict = {}

    for key, value in flat_dict.items():
        keys = key.split(separator)
        current_dict = parent_dict
        for i, k in enumerate(keys[:-1]):
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value

    return parent_dict


class ClusterInformation:
    def __init__(self, hosts, debug=False):
        self.debug = debug
        self.hosts = hosts
        import pdb
        pdb.set_trace()
        self.info = self.collect_info()

    def __str__(self):
        return pprint.pformat(self.info)

    def __repr__(self):
        return str(self.info)

    def collect_info(self):
        info = {}

        if self.debug:
            print(f"Collecting information for hosts: {self.hosts}")
            return {host: {'cpu_core_count': 0,'memory_info': {}} for host in self.hosts}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_host = {executor.submit(self.get_host_info, host): host for host in self.hosts}
            for future in concurrent.futures.as_completed(future_to_host):
                host = future_to_host[future]
                cpu_core_count, memory_info = future.result()
                info[host] = {
                    'cpu_core_count': cpu_core_count,
                    'memory_info': memory_info
                }
        return info

    def get_host_info(self, host):
        if host in ['localhost', '127.0.0.1']:
            return self.get_local_info()
        else:
            return self.get_remote_info(host)

    def get_local_info(self):
        cpu_core_count = psutil.cpu_count(logical=False)
        memory_info = dict(psutil.virtual_memory()._asdict())
        return cpu_core_count, memory_info

    def get_remote_info(self, host):
        cpu_core_count = self.get_remote_cpu_core_count(host)
        memory_info = self.get_remote_memory_info(host)
        return cpu_core_count, memory_info

    @staticmethod
    def get_remote_cpu_core_count(host):
        cpu_core_count = 0
        cpu_info_path = f"ssh {host} cat /proc/cpuinfo"
        try:
            output = os.popen(cpu_info_path).read()
            cpu_core_count = output.count('processor')
        except Exception as e:
            print(f"Error getting CPU core count for host {host}: {e}")
        return cpu_core_count

    @staticmethod
    def get_remote_memory_info(host):
        memory_info = {}
        meminfo_path = f"ssh {host} cat /proc/meminfo"
        try:
            output = os.popen(meminfo_path).read()
            lines = output.split('\n')
            for line in lines:
                if line.startswith('MemTotal:'):
                    memory_info['total'] = int(line.split()[1])
                elif line.startswith('MemFree:'):
                    memory_info['free'] = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    memory_info['available'] = int(line.split()[1])
        except Exception as e:
            print(f"Error getting memory information for host {host}: {e}")
        return memory_info
