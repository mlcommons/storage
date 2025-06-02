# MLPerf™ Storage Benchmark Suite Setup (Ansible Playbook)

This repository contains an Ansible playbook to help automate the setup of the MLPerf™ Storage Benchmark Suite using Bare-Metal for different OS (Ubuntu, Red Hat, SLES).

## 📋 Prerequisites

- Ensure `ansible_user` has **sudo** privileges.
- It's recommended to set up **password-less SSH and sudo** for smooth execution.
- **Python**: Version 3.9  
- **Ansible**: Version 2.10+  
  
### Set up password-less SSH:
```bash
ssh-keygen
ssh-copy-id root@172.22.X.X
```

## 📦 Installation

### Install Python dependencies:
```bash
python3 -m pip install ansible ansible-pylibssh
```

## 🗂️ Preparing Mount Directories

Ensure your dataset folders exist before running benchmarks on **every** server.

### Option 1: Local Directory

Create local folders:
```bash
mkdir -p /mnt/nfs/train /mnt/nfs/valid
```

### Option 2: NFS Mounted Directory

Use `/mnt/nfs` as mount location.

## 🚀 Running Bare-Metal Ansible Setup

### 1. Update the inventory file with the target IPs:
```bash
nano inventory
```

### 2. Run the playbook:
```bash
cd mlcommons-storage/ansible/
ansible-playbook -i inventory setup.yml
```

### 3. Activate virtual environment and prepare data
```bash
cd storage/
source venv/bin/activate
```

### 4. Run benchmark:
**Data generation:**
```bash
mlpstorage training datagen --hosts 172.X.X.1 172.X.X.2 --num-processes 8 --model cosmoflow --data-dir /mnt/nfs/data --results-dir /mnt/nfs/result --param dataset.num_files_train=100
```
**Benchmark Run:**
```bash
mlpstorage training run --hosts 172.X.X.1 172.X.X.2  --num-client-hosts 2 --client-host-memory-in-gb 64 --num-accelerators 8 --accelerator-type h100 --model cosmoflow --data-dir /mnt/nfs/data --results-dir /mnt/nfs/result --param dataset.num_files_train=100
