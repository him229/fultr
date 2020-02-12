# MSLR Dataset
dataset="mslr"
root_directory="transformed_datasets/mslr"
noise_root_directory="transformed_datasets/mslr-noise-0.1"
noise_level="0.1"
train_sizes="'359','1k','4k','12k','36k','120k'"
partial_size="12k"

# German Credit Dataset
dataset="german"
root_directory="transformed_datasets/german"
noise_root_directory="transformed_datasets/german-noise-0.1"
noise_level="0.1"
train_sizes="'50','250','500','2k','5k','25k','50k','250k'"
partial_size="5k"

# Our method (linear model)
python my_dispatcher.py ${dataset} --root_directory ${root_directory} --no-gpu --comment square --train_sizes "['full','${partial_size}'']"
# Our method (MLP)
python my_dispatcher.py ${dataset} --root_directory ${root_directory} --mlp --no-gpu --comment square --train_sizes "['full','${partial_size}']"

# Baselines
# Colorblind
python my_dispatcher.py ${dataset} --root_directory ${root_directory} --masked --lambdas "[0.0]" --no-gpu --train_sizes "['${partial_size}']"
# Ashudeep
python my_dispatcher.py ${dataset} --root_directory ${root_directory} --ashudeep --no-gpu --train_sizes "['${partial_size}']"

# Abalation Study
python my_dispatcher.py ${dataset} --root_directory ${root_directory} --unweighted --no-gpu --comment square --train_sizes "['${partial_size}']"
python my_dispatcher.py ${dataset} --root_directory ${root_directory} --unweighted_fairness --no-gpu --comment square --train_sizes "['${partial_size}']"

# Noise experiment
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --no-gpu --comment square-sample --train_sizes "['36k']" --train_folder Train-sample
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --noise --no-gpu --comment square --train_sizes "['36k']"
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --noise --no-gpu --comment square --train_sizes "['120k']" --memory_limit=5

# Utility for Ashudeep
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --no-gpu --utility --train_sizes "[${train_sizes}]" --lambdas "[0.0]" --unweighted --memory_limit=5

# Utility for our method
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --no-gpu --utility --train_sizes "[${train_sizes}]" --lambdas "[0.0]" --memory_limit=5
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --no-gpu --utility --train_sizes "[${train_sizes}]" --lambdas "[0.0]" --mlp --memory_limit=5

# Skyline
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --no-gpu --utility --skyline --train_sizes "['full']" --lambdas "[0.0]" --memory_limit=5
python my_dispatcher.py ${dataset}-noise-${noise_level} --root_directory ${noise_root_directory} --no-gpu --utility --skyline --train_sizes "['full']" --lambdas "[0.0]" --mlp --memory_limit=5