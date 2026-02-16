Creating a Checkpointing Plugin
==============================

Within DLIO Benchmark we can define custom checkpointing implementations.
This feature allows us to extend DLIO Benchmark with new checkpointing implementation easily without changing existing code.
To achieve this developers have to take the following main steps.

1. Write their custom checkpointing.
2. Define workflow configuration.
3. Run the workload with custom checkpointing.

Write their custom checkpointing.
--------------------------------

In this section, we will describe how to write the custom checkpointing.
To write a checkpointing you need to implement `BaseCheckpointing` Class.
This checkpointing needs to added `<ROOT>/dlio_benchmark/plugins/experimental/src/checkpointing`.
A complete examples can be seen at `<ROOT>/dlio_benchmark/checkpointing/`

- For PyTorch: pytorch_checkpointing.py
- For TensorFlow: tf_checkpointing.py
  
Say we store the custom checkpointing for pytorch into `<ROOT>/dlio_benchmark/plugins/experimental/src/checkpoint/pytorch_checkpointing.py`

.. code-block:: python

    class CustomPyTorchCheckpointing(BaseCheckpointing):
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if CustomPyTorchCheckpointing.__instance is None:
            CustomPyTorchCheckpointing.__instance = CustomPyTorchCheckpointing()
        return CustomPyTorchCheckpointing.__instance

    @dlp.log_init
    def __init__(self):
        super().__init__("pt")

    @dlp.log
    def get_tensor(self, size):
        return torch.randint(high=1, size=(size,), dtype=torch.int8)

    @dlp.log
    def save_state(self, suffix, state):
        name = self.get_name(suffix)
        with open(name, "wb") as f:
            torch.save(state, f)

    @dlp.log
    def checkpoint(self, epoch, step_number):
        super().checkpoint(epoch, step_number)

Define workflow configuration.
------------------------------

In this section, we will detail how to create a custom workflow configuration for DLIO Benchmark.
The workload configuration for plugins exists in `<ROOT>/dlio_benchmark/plugins/experimental`.
You can copy an existing configuration from `<ROOT>/dlio_benchmark/configs/workload` and modify it for your custom checkpointing.
Main changes to the workflow configuration are:

.. code-block:: yaml

    # Rest remains as it is
    reader:
          checkpoint_mechanism_classname: dlio_benchmark.plugins.experimental.src.checkpoint.pytorch_checkpointing.CustomPyTorchCheckpointing


In the above configuration, `checkpoint_mechanism_classname` should point to FQN of the class (as in the PYTHONPATH).


Run the workload with custom checkpointing.
------------------------------------------

To run the custom checkpointing, we have to define the plugin folder as the custom config folder.
This is described in the :ref:`run` page. 
We need to pass path `plugins/experimental/configs` as the path.