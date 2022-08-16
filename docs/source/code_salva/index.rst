.. _aibedo_code


.. automodule:: aibedo.models

Base AiBEDO class
--------------------

.. autoclass:: BaseModel

.. autosummary::
    :toctree: generated
    :nosignatures:

    BaseModel.forward
    BaseModel.raw_predict
    BaseModel.raw_outputs_to_denormalized_per_variable_dict
    BaseModel.postprocess_raw_predictions
    BaseModel.predict

.. automodule:: aibedo.models.MLP
    :members:
    :undoc-members: [train_step_initial_log_dict]
    :show-inheritance: False

.. automodule:: aibedo

Training and Testing
--------------------

The following are the main functions that you want to use for training and/or testing
the AiBEDO models.

.. autosummary::
    :toctree: generated
    :nosignatures:

    train.run_model
    test.reload_and_test_model



Interface
----------

The main
:func:`training <aibedo.train.run_model>` and
:func:`testing <aibedo.test.reload_and_test_model>` scripts above,
calls various helper functions to avoid model/data loading (and reloading) boilerplate code.
If the main training/testing scripts above are not enough for your purposes,
we strongly recommend using the interface functions below as much as possible.

.. autosummary::
    :toctree: generated
    :nosignatures:

    interface.get_model
    interface.get_datamodule
    interface.get_model_and_data
    interface.reload_model_from_config_and_ckpt

