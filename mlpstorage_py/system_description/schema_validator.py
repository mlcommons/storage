# Pydantic v2 schema validator for MLPerf Storage system description YAML files.
# Requires: pip install pydantic>=2.0  (add to pyproject.toml dependencies)
#
# Programmatic usage:
#   from mlpstorage_py.system_description.schema_validator import validate_file, validate_dict
#   errors = validate_file("path/to/system.yaml")
#   if errors:
#       for msg in errors:
#           print(msg)
#
# CLI usage:
#   python schema_validator.py <yaml-file> [yaml-file ...]

import sys
from enum import Enum
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StorageLocation(str, Enum):
    remote           = 'remote'
    local            = 'local'
    remote_and_local = 'remote_and_local'


class BenchmarkAPI(str, Enum):
    file   = 'file'
    object = 'object'


class ProductAPI(str, Enum):
    file   = 'file'
    object = 'object'
    block  = 'block'


class ClientFootprint(str, Enum):
    open_source   = 'open_source'
    closed_source = 'closed_source'
    na            = 'n/a'


class ClientInstallation(str, Enum):
    in_box      = 'in_box'
    installable = 'installable'
    na          = 'n/a'


class DeploymentMode(str, Enum):
    onprem = 'onprem'
    cloud  = 'cloud'


class NetworkType(str, Enum):
    ethernet   = 'ethernet'
    infiniband = 'infiniband'
    other      = 'other'


class TrafficType(str, Enum):
    management = 'management'
    data       = 'data'
    metadata   = 'metadata'
    backend    = 'backend'


class DriveInterface(str, Enum):
    nvme  = 'nvme'
    SAS   = 'SAS'
    SATA  = 'SATA'
    other = 'other'


class PowerEfficiency(str, Enum):
    Standard = 'Standard'
    Bronze   = 'Bronze'
    Silver   = 'Silver'
    Gold     = 'Gold'
    Platinum = 'Platinum'
    Titanium = 'Titanium'
    Ruby     = 'Ruby'
    Unrated  = 'Unrated'


# ---------------------------------------------------------------------------
# Leaf models
# ---------------------------------------------------------------------------

class KeyValue(BaseModel):
    """A single name/value tuning pair (OS environment variable, sysctl knob, etc.)."""
    name:  str = Field(min_length=1)
    value: str = Field(min_length=1)


class PowerSupply(BaseModel):
    unit_count:           int            = Field(ge=1)
    inlet_voltage:        int            = Field(gt=0)
    nameplate_power_watts: int           = Field(ge=1)
    efficiency:           PowerEfficiency


class PowerDevice(BaseModel):
    """Power configuration for a chassis or switch (PSU count and redundancy model)."""
    min_psus_active:  int               = Field(ge=1)
    psus_configured:  List[PowerSupply] = Field(min_length=1)

    @model_validator(mode='after')
    def check_psu_count(self) -> 'PowerDevice':
        # Rule 12: min_psus_active must not exceed total installed PSU count
        total = sum(psu.unit_count for psu in self.psus_configured)
        if self.min_psus_active > total:
            raise ValueError(
                f"min_psus_active ({self.min_psus_active}) exceeds total installed "
                f"PSU count ({total}) across all psus_configured entries"
            )
        return self


class NetworkPort(BaseModel):
    """One homogeneous group of NIC ports sharing the same type, speed, and traffic role."""
    unit_count:  int               = Field(ge=1)
    type:        NetworkType
    speed:       int               = Field(ge=1)   # Gigabits/s
    traffic:     List[TrafficType] = Field(min_length=1)


class DriveInstance(BaseModel):
    """One homogeneous group of storage drives sharing the same model and configuration."""
    unit_count:      int            = Field(ge=1)
    vendor_name:     str            = Field(min_length=1)
    model_name:      str            = Field(min_length=1)
    interface:       DriveInterface
    capacity_in_GB:  int            = Field(ge=1)  # base-10 GB


# ---------------------------------------------------------------------------
# Mid-level models
# ---------------------------------------------------------------------------

class OperatingSystem(BaseModel):
    name:    str = Field(min_length=1)
    version: str = Field(min_length=1)


class Chassis(BaseModel):
    model_name:       str                  = Field(min_length=1)
    rack_units:       Optional[int]        = Field(default=None, ge=1)
    cpu_model:        str                  = Field(min_length=1)
    cpu_qty:          int                  = Field(ge=1)
    cpu_cores:        int                  = Field(ge=1)
    memory_capacity:  int                  = Field(ge=1)   # GiB
    power:            Optional[PowerDevice] = None


class NodeDescription(BaseModel):
    """Describes a homogeneous group of nodes (product nodes or benchmark clients)."""
    friendly_description:  str                      = Field(min_length=1)
    quantity:              int                      = Field(ge=1)
    chassis:               Chassis
    networking:            Optional[List[NetworkPort]]   = None
    drives:                Optional[List[DriveInstance]] = None
    operating_system:      OperatingSystem
    environment:           Optional[List[KeyValue]]      = None
    sysctl:                Optional[List[KeyValue]]      = None


class SwitchDescription(BaseModel):
    """Describes a homogeneous group of network switches included in the storage solution."""
    unit_count:    int                      = Field(ge=1)
    vendor_name:   str                      = Field(min_length=1)
    model_name:    str                      = Field(min_length=1)
    ports:         List[NetworkPort]        = Field(min_length=1)
    configuration: Optional[List[KeyValue]] = None
    rack_units:    int                      = Field(ge=1)
    power:         Optional[PowerDevice]    = None


# ---------------------------------------------------------------------------
# Top-level solution models
# ---------------------------------------------------------------------------

class Architecture(BaseModel):
    storage_location:    StorageLocation
    benchmark_API:       BenchmarkAPI
    product_API:         ProductAPI
    client_footprint:    ClientFootprint
    client_installation: ClientInstallation

    @model_validator(mode='after')
    def check_na_pairing(self) -> 'Architecture':
        # Rule 14: client_footprint and client_installation must both be 'n/a' or neither
        fp_na = self.client_footprint == ClientFootprint.na
        ci_na = self.client_installation == ClientInstallation.na
        if fp_na != ci_na:
            raise ValueError(
                "client_footprint and client_installation must both be 'n/a' or neither; "
                f"got '{self.client_footprint.value}' and '{self.client_installation.value}'"
            )
        return self


class Capabilities(BaseModel):
    multi_host:             bool
    simultaneous_write:     bool
    simultaneous_read:      bool
    remap_time_in_seconds:  int = Field(ge=0)

    @model_validator(mode='after')
    def check_remap_time(self) -> 'Capabilities':
        # Rule 13: remap_time_in_seconds == 0  iff  both simultaneous flags are true
        both_simultaneous = self.simultaneous_write and self.simultaneous_read
        if both_simultaneous and self.remap_time_in_seconds != 0:
            raise ValueError(
                f"remap_time_in_seconds must be 0 when both simultaneous_write and "
                f"simultaneous_read are true (got {self.remap_time_in_seconds})"
            )
        if not both_simultaneous and self.remap_time_in_seconds == 0:
            raise ValueError(
                "remap_time_in_seconds must be non-zero when either simultaneous_write "
                f"or simultaneous_read is false (simultaneous_write={self.simultaneous_write}, "
                f"simultaneous_read={self.simultaneous_read})"
            )
        return self


class Solution(BaseModel):
    submission_name:      str          = Field(min_length=1)
    friendly_description: str          = Field(min_length=1)
    architecture:         Architecture
    capabilities:         Capabilities


class SystemUnderTest(BaseModel):
    solution:             Solution
    deployment:           DeploymentMode
    product_nodes:        Optional[List[NodeDescription]]   = None
    product_switches:     Optional[List[SwitchDescription]] = None
    total_rack_units:     Optional[int]                     = Field(default=None, ge=0)
    rack_power_supplies:  Optional[List[PowerDevice]]       = None
    clients:              List[NodeDescription]

    @model_validator(mode='after')
    def check_deployment_rules(self) -> 'SystemUnderTest':
        # Rules 1–6 (onprem) and 7–10 (cloud)
        if self.deployment == DeploymentMode.onprem:
            for i, node in enumerate(self.product_nodes or []):
                if node.chassis.rack_units is None:
                    raise ValueError(f"product_nodes[{i}].chassis.rack_units is required for onprem deployment")
                if node.chassis.power is None:
                    raise ValueError(f"product_nodes[{i}].chassis.power is required for onprem deployment")
            for i, client in enumerate(self.clients):
                if client.chassis.rack_units is None:
                    raise ValueError(f"clients[{i}].chassis.rack_units is required for onprem deployment")
                if client.chassis.power is None:
                    raise ValueError(f"clients[{i}].chassis.power is required for onprem deployment")
            if self.total_rack_units is None:
                raise ValueError("total_rack_units is required for onprem deployment")
            for i, switch in enumerate(self.product_switches or []):
                if switch.rack_units is None:
                    raise ValueError(f"product_switches[{i}].rack_units is required for onprem deployment")
                if switch.power is None:
                    raise ValueError(f"product_switches[{i}].power is required for onprem deployment")

        elif self.deployment == DeploymentMode.cloud:
            if self.product_switches:
                raise ValueError("product_switches must be absent for cloud deployment")
            if self.rack_power_supplies:
                raise ValueError("rack_power_supplies must be absent for cloud deployment")
            for i, node in enumerate(self.product_nodes or []):
                if node.chassis.rack_units is not None:
                    raise ValueError(f"product_nodes[{i}].chassis.rack_units must be absent for cloud deployment")
                if node.chassis.power is not None:
                    raise ValueError(f"product_nodes[{i}].chassis.power must be absent for cloud deployment")
            for i, client in enumerate(self.clients):
                if client.chassis.rack_units is not None:
                    raise ValueError(f"clients[{i}].chassis.rack_units must be absent for cloud deployment")
                if client.chassis.power is not None:
                    raise ValueError(f"clients[{i}].chassis.power must be absent for cloud deployment")
        return self

    @model_validator(mode='after')
    def check_rack_units_sum(self) -> 'SystemUnderTest':
        # Rule 11: total_rack_units == sum of product_nodes RUs + product_switches RUs (clients excluded)
        if self.total_rack_units is None:
            return self
        node_rus = sum(
            node.chassis.rack_units * node.quantity
            for node in (self.product_nodes or [])
            if node.chassis.rack_units is not None
        )
        switch_rus = sum(
            switch.rack_units * switch.unit_count
            for switch in (self.product_switches or [])
        )
        expected = node_rus + switch_rus
        if self.total_rack_units != expected:
            raise ValueError(
                f"total_rack_units ({self.total_rack_units}) does not match the sum of "
                f"product_nodes rack units ({node_rus}) + product_switches rack units "
                f"({switch_rus}) = {expected}  (clients are excluded from this sum)"
            )
        return self

    @model_validator(mode='after')
    def check_networking_rules(self) -> 'SystemUnderTest':
        # Rules 15–17: networking requirements driven by storage_location
        storage_loc = self.solution.architecture.storage_location
        needs_networking = storage_loc in (StorageLocation.remote, StorageLocation.remote_and_local)
        forbid_switches  = storage_loc == StorageLocation.local

        if needs_networking:
            for i, node in enumerate(self.product_nodes or []):
                if not node.networking:
                    raise ValueError(
                        f"product_nodes[{i}].networking is required when "
                        f"storage_location is '{storage_loc.value}'"
                    )
            for i, client in enumerate(self.clients):
                if not client.networking:
                    raise ValueError(
                        f"clients[{i}].networking is required when "
                        f"storage_location is '{storage_loc.value}'"
                    )

        if forbid_switches and self.product_switches:
            raise ValueError("product_switches must be absent when storage_location is 'local'")
        return self


class SystemDescription(BaseModel):
    """Root model — the entire system_under_test document."""
    system_under_test: SystemUnderTest


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_dict(data: dict) -> list[str]:
    """Validate a pre-loaded dict. Returns human-readable error strings; empty list means valid."""
    try:
        SystemDescription.model_validate(data)
        return []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return errors


def validate_file(path: str | Path) -> list[str]:
    """Load a YAML file and validate it. Returns human-readable error strings; empty list means valid."""
    path = Path(path)
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]
    except OSError as exc:
        return [f"Cannot read file: {exc}"]

    if not isinstance(data, dict):
        return ["Top-level YAML value must be a mapping"]

    return validate_dict(data)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <yaml-file> [yaml-file ...]")
        sys.exit(1)

    any_failures = False
    for arg in sys.argv[1:]:
        errors = validate_file(arg)
        if errors:
            print(f"FAIL  {arg}")
            for msg in errors:
                print(f"      {msg}")
            any_failures = True
        else:
            print(f"OK    {arg}")

    sys.exit(1 if any_failures else 0)
