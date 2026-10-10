====================
实验运行工具
====================

.. contents:: 目录

ExperimentGrid（实验网格）
==============================

Spinning Up 提供了一个名为 ``ExperimentGrid`` 的工具，使超参数消融实验更容易开展。它借鉴了 rllab 中名为 ``VariantGenerator`` 的 `rllab 工具`_，但比后者更简单。

.. _`rllab 工具`: https://github.com/rll/rllab/blob/master/rllab/misc/instrument.py#L173

.. autoclass:: spinup.utils.run_utils.ExperimentGrid
    :members:


调用实验
===================

.. autofunction:: spinup.utils.run_utils.call_experiment

.. autofunction:: spinup.utils.run_utils.setup_logger_kwargs
