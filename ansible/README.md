# What is this?

This is an ansible playbook for helping set up an MLPerf™ Storage Benchmark Suite.

## How do you use this?

The ansible_user should have sudo privileges.  Preferably, set up password-less
ssh and password-less sudo for that user, though you can instead instruct
ansible to ask for passwords using `-k, --ask-pass` from [docs](https://docs.ansible.com/ansible/latest/cli/ansible-playbook.html#cmdoption-ansible-playbook-k).

This is an example how to set password-less:

```bash
 ssh-copy-id root@172.22.X.X
```

Install dependencies:

```bash
python3 -m pip install ansible-pylibssh
ansible-galaxy collection install -r collections/requirements.yml
```

Then run the playbook:

```bash
ansible-playbook -i inventory setup.yml
```

Run the playbook only on specific host or group:

```bash
ansible-playbook -i inventory -l worker1 setup.yml
```

## Example log

```bash
root@management:~/mlcommons/storage/ansible# ansible-playbook -i inventory setup.yml

TBD...
```
