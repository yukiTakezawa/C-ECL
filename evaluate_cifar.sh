lr=0.005

itr_per_round=5
alpha=12.5
epoch=2500

mkdir results
mkdir results/cifar_lenet


for class in {8,10} ; do
    mkdir results/cifar_lenet/class_${class}_itr_${itr_per_round}

    theta=1.0
    for comp_rate in {0.01,0.1,0.2} ; do
	log_path=./results/cifar_lenet/class_${class}_itr_${itr_per_round}/cecl_${comp_rate}_lr_${lr}/
	mkdir ${log_path}
	python evaluate_cifar.py cecl ${log_path} --port 1579067 --nw config/ring_class${class}_2.json --lr ${lr} --itr_per_round ${itr_per_round} --epoch ${epoch} --theta ${theta}  --comp_rate ${comp_rate}
    done
done
