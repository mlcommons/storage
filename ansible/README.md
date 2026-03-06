# MLPerf™ Storage Benchmark Suite Setup (Ansible Playbook)

This repository contains an Ansible playbook to automate the setup of the **MLPerf™ Storage Benchmark Suite** on **bare-metal nodes** across multiple operating systems (Ubuntu, Red Hat, SLES).

## 📋 Prerequisites

- **One node must act as the control node** — this is where Ansible is installed and run.
  All other nodes are *target nodes* and **do not need Ansible installed**.
- The control node must be able to SSH into all target nodes.
- Ensure the `ansible_user` has **sudo** privileges on all nodes.
- Set up **password-less SSH and sudo** to avoid authentication interruptions.
  
### Set up password-less SSH (on control node):

```bash
ssh-keygen
ssh-copy-id root@172.22.X.X  # Run for each target node
```

## 📦 Installation

### Install Python dependencies:

```bash
uv venv 
source venv/bin/activate
uv pip install ansible ansible-pylibssh
ansible-galaxy collection install -r collections/requirements.yml
```

## 🗂️ Preparing Mount Directories

Before running benchmarks, **ensure dataset folders exist** on **every node**.

### Option 1: Local Directory

Create local folders:

```bash
mkdir -p /mnt/nfs/train /mnt/nfs/valid
```

### Option 2: NFS Mounted Directory

Use `/mnt/nfs` as a shared mount location across nodes.

```bash
sudo apt install nfs-common  # Or use zypper/yum based on OS
sudo mkdir -p /mnt/nfs
sudo mount -t nfs 172.22.X.X:/shared-path /mnt/nfs
```

To persist it, add to `/etc/fstab`:

```bash
172.22.X.X:/shared-path /mnt/nfs nfs defaults 0 0
```

## 🚀 Running Bare-Metal Ansible Setup

### 1. Update the inventory file with the target IPs:

```bash
nano inventory
```

### 2. Run the playbook:

```bash
cd storage/ansible/
ansible-playbook -i inventory setup.yml
```

### 3. Activate virtual environment

```bash
source venv/bin/activate
```

### 4. Data Generation

This step should run only once per model, as data generation is time-consuming.

```bash
mlpstorage training datagen --hosts 172.X.X.1 172.X.X.2 --num-processes 8 --model cosmoflow --data-dir /mnt/nfs/data --results-dir /mnt/nfs/result --param dataset.num_files_train=100
```

### 5. Run training Benchmark

```bash
mlpstorage training run --hosts 172.X.X.1 172.X.X.2  --num-client-hosts 2 --client-host-memory-in-gb 64 --num-accelerators 8 --accelerator-type h100 --model cosmoflow --data-dir /mnt/nfs/data --results-dir /mnt/nfs/result --param dataset.num_files_train=100
```