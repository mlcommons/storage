#from distutils import util
import sysconfig
from setuptools import find_namespace_packages, setup
import pathlib

HYDRA_VERSION = "1.3.2"

test_deps = [
    "pytest",
    "pytest-xdist",
    "dftracer>=2.0.1",
]
core_deps = [
    "Pillow>=9.3.0",
    "PyYAML>=6.0.0",
    "h5py>=3.11.0",
    "mpi4py>=3.1.4",
    "numpy>=1.23.5",
    "omegaconf>=2.2.0",
    "pandas>=1.5.1",
    "psutil>=5.9.8",
    "pydftracer>=2.0.2"
]
x86_deps = [
    f"hydra-core>={HYDRA_VERSION}",
    "nvidia-dali-cuda120>=1.34.0",
    "tensorflow>=2.13.1",
    "torch>=2.2.0",
    "torchaudio",
    "torchvision",
]
ppc_deps = [
    f"hydra-core @ git+https://github.com/facebookresearch/hydra.git@v{HYDRA_VERSION}#egg=hydra-core"
]

deps = core_deps

if "ppc" in sysconfig.get_platform():
    deps.extend(ppc_deps)
else:
    deps.extend(x86_deps)

extras = {
    "test": test_deps,
    "dftracer": [
        "dftracer>=2.0.1",
    ],
    "s3": [
        "s3torchconnector",
    ],
}

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="dlio_benchmark",
    version="2.0.0",
    description="An I/O benchmark for deep learning applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/argonne-lcf/dlio_benchmark",
    author="Huihuo Zheng, Hariharan Devarajan (Hari)",
    author_email="zhenghh04@gmail.com, mani.hariharan@gmail.com",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="deep learning, I/O, benchmark, NPZ, pytorch benchmark, tensorflow benchmark",
    project_urls={  # Optional
        "Documentation": "https://dlio-benchmark.readthedocs.io",
        "Source": "https://github.com/argonne-lcf/dlio_benchmark",
        "Release Notes": "https://github.com/argonne-lcf/dlio_benchmark/releases",
        "Bug Reports": "https://github.com/argonne-lcf/dlio_benchmark/issues",
    },
    # Main package definition
    packages=find_namespace_packages(where="."),
    package_dir={"dlio_benchmark": "dlio_benchmark"},
    package_data={
        "dlio_benchmark.configs": ["*.yaml"],
        "dlio_benchmark.configs.hydra.help": ["*.yaml"],
        "dlio_benchmark.configs.hydra.job_logging": ["*.yaml"],
        "dlio_benchmark.configs.workload": ["*.yaml"],
    },
    dependency_links=[
        "https://download.pytorch.org/whl/cpu",
        "https://developer.download.nvidia.com/compute/redist",
    ],
    install_requires=deps,
    tests_require=test_deps,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "dlio_benchmark = dlio_benchmark.main:main",
            "dlio_benchmark_query = dlio_benchmark.main:query_config",
            "dlio_postprocessor = dlio_benchmark.postprocessor:main",
        ]
    },
)
