python main.py  --purpose Cifar --device cuda:2 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5
python ../main.py  --purpose Cifar --device cuda:3 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python ../main.py  --purpose Cifar --device cuda:2 --global_seed 0 --use_wandb False --yamlfile ./Cifar100_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &
python ../main.py  --purpose Cifar --device cuda:3 --global_seed 0 --use_wandb False --yamlfile ./Cifar100_Conv2Cifar.yaml --strategy FedNH --num_clients 100 --participate_ratio 0.1 --partition noniid-label-distribution --beta 1.0 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5 &

python main.py  --purpose Cifar --device cuda:2 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 100 
--participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 
--sgd_weight_decay 1e-05 --num_epochs 5

# 10
python main.py  --purpose Braintumor --device cuda:1 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 10 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5
# 40
python main.py  --purpose Braintumor --device cuda:1 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 40 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5
# 50
python main.py  --purpose Braintumor --device cuda:1 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 50 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5

# 60
python main.py  --purpose Braintumor --device cuda:1 --global_seed 0 --use_wandb False --yamlfile ./Cifar10_Conv2Cifar.yaml --strategy FedNH --num_clients 60 --participate_ratio 0.1 --partition noniid-label-distribution --beta 0.3 --num_rounds 200 --client_lr 0.01 --client_lr_scheduler diminishing --sgd_momentum 0.9 --sgd_weight_decay 1e-05 --num_epochs 5