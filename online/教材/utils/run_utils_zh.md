---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/utils/run_utils_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/utils/run_utils_zh.rst
---
# 实验运行工具



## ExperimentGrid（实验网格）

Spinning Up 提供了一个名为 `ExperimentGrid` 的工具，使超参数消融实验更容易开展。它借鉴了 rllab 中名为 `VariantGenerator` 的 [rllab 工具](https://github.com/rll/rllab/blob/master/rllab/misc/instrument.py#L173)，但比后者更简单。

### API：spinup.utils.run_utils.ExperimentGrid

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/run_utils.py

#### ExperimentGrid

Tool for running many experiments given hyperparameter ranges.

#### ExperimentGrid.__init__(self, name='')

#### ExperimentGrid.name(self, _name)

#### ExperimentGrid.print(self)

Print a helpful report about the experiment grid.

#### ExperimentGrid.add(self, key, vals, shorthand=None, in_name=False)

Add a parameter (key) to the grid config, with potential values (vals).

By default, if a shorthand isn't given, one is automatically generated from the key using the first three letters of each colon-separated term. To disable this behavior, change `DEFAULT_SHORTHAND` in the `spinup/user_config.py` file to `False`.

Args:  
key (string): Name of parameter.

vals (value or list of values): Allowed values of parameter.

shorthand (string): Optional, shortened name of parameter. For  
example, maybe the parameter `steps_per_epoch` is shortened to `steps`.

in_name (bool): When constructing variant names, force the  
inclusion of this parameter into the name.

#### ExperimentGrid.variant_name(self, variant)

Given a variant (dict of valid param/value pairs), make an exp_name.

A variant's name is constructed as the grid name (if you've given it one), plus param names (or shorthands if available) and values separated by underscores.

Note: if `seed` is a parameter, it is not included in the name.

#### ExperimentGrid.variants(self)

Makes a list of dicts, where each dict is a valid config in the grid.

There is special handling for variant parameters whose names take the form

> `'full:param:name'`.

The colons are taken to indicate that these parameters should have a nested dict structure. eg, if there are two params,

> | Key              | Val |
> |------------------|-----|
> | `'base:param:a'` | 1   |
> | `'base:param:b'` | 2   |

the variant dict will have the structure


```text
variant = {  
base: {  
param : {  
a : 1, b : 2 }

}

}
```


#### ExperimentGrid.run(self, thunk, num_cpu=1, data_dir=None, datestamp=False)

Run each variant in the grid with function 'thunk'.

Note: 'thunk' must be either a callable function, or a string. If it is a string, it must be the name of a parameter whose values are all callable functions.

Uses `call_experiment` to actually launch each experiment, and gives each variant a name using `self.variant_name()`.

Maintenance note: the args for ExperimentGrid.run should track closely to the args for call_experiment. However, `seed` is omitted because we presume the user may add it as a parameter in the grid.


## 调用实验

### API：spinup.utils.run_utils.call_experiment

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/run_utils.py

#### call_experiment(exp_name, thunk, seed=0, num_cpu=1, data_dir=None, datestamp=False, **kwargs)

Run a function (thunk) with hyperparameters (kwargs), plus configuration.

This wraps a few pieces of functionality which are useful when you want to run many experiments in sequence, including logger configuration and splitting into multiple processes for MPI.

There's also a SpinningUp-specific convenience added into executing the thunk: if `env_name` is one of the kwargs passed to call_experiment, it's assumed that the thunk accepts an argument called `env_fn`, and that the `env_fn` should make a gym environment with the given `env_name`.

The way the experiment is actually executed is slightly complicated: the function is serialized to a string, and then `run_entrypoint.py` is executed in a subprocess call with the serialized string as an argument. `run_entrypoint.py` unserializes the function call and executes it. We choose to do it this way---instead of just calling the function directly here---to avoid leaking state between successive experiments.

Args:

> exp_name (string): Name for experiment.
>
> thunk (callable): A python function.
>
> seed (int): Seed for random number generators.
>
> num_cpu (int): Number of MPI processes to split into. Also accepts  
> 'auto', which will set up as many procs as there are cpus on the machine.
>
> data_dir (string): Used in configuring the logger, to decide where  
> to store experiment results. Note: if left as None, data_dir will default to `DEFAULT_DATA_DIR` from `spinup/user_config.py`.
>
> \*\*kwargs: All kwargs to pass to thunk.


### API：spinup.utils.run_utils.setup_logger_kwargs

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/run_utils.py

#### setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False)

Sets up the output_dir for a logger and returns a dict for logger kwargs.

If no seed is given and datestamp is false,

    output_dir = data_dir/exp_name

If a seed is given and datestamp is false,

    output_dir = data_dir/exp_name/exp_name_s[seed]

If datestamp is true, amend to

    output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]

You can force datestamp=True by setting `FORCE_DATESTAMP=True` in `spinup/user_config.py`.

Args:

> exp_name (string): Name for experiment.
>
> seed (int): Seed for random number generators used by experiment.
>
> data_dir (string): Path to folder where results should be saved.  
> Default is the `DEFAULT_DATA_DIR` in `spinup/user_config.py`.
>
> datestamp (bool): Whether to include a date and timestamp in the  
> name of the save directory.

Returns:

> logger_kwargs, a dict containing output_dir and exp_name.
