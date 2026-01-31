echo "Start Train"
dir=$(dirname "$0")
if [ -f "$dir/Train.py" ];then
    cd $dir
    pwd
    export CUDA_VISIBLE_DEVICES=0
    ##### run with Polyp Datasets
    python Train.py --model_name MDFANet --epoch 151 --batchsize 16 --trainsize 256 --train_save MDFANet_Kvasir_1e4_bs16_e150_s256 --lr 0.0001 --train_path /root/autodl-tmp/MDFANet/data/TrainDataset --test_path /root/autodl-tmp/MDFANet/data/TestDataset/Kvasir/
    sleep 1m
    python Test.py --train_save MDFANet_Kvasir_1e4_bs16_e150_s256 --testsize 256 --test_path /root/autodl-tmp/MDFANet/data/TestDataset
    
    ### run with ISIC 2018
    python Train.py --model_name MDFANet --epoch 151 --batchsize 16 --trainsize 256 --train_save MDFANet_ISIC2018_1e4_bs16_e150_s256 --lr 0.0001 --train_path /root/autodl-tmp/MDFANet/data/ISIC2018/train --test_path /root/autodl-tmp/MDFANet/data/ISIC2018/val/
    sleep 1m
    python Test.py --train_save MDFANet_ISIC2018_1e4_bs16_e150_s256 --testsize 256 --test_path /root/autodl-tmp/MDFANet/data/ISIC2018

    ### run with ISIC 2017
    python Train.py --model_name MDFANet --epoch 151 --batchsize 16 --trainsize 256 --train_save MDFANet_ISIC2017_1e4_bs16_e150_s256 --lr 0.0001 --train_path /root/autodl-tmp/MDFANet/data/ISIC2017/train --test_path /root/autodl-tmp/MDFANet/data/ISIC2017/val/
    sleep 1m
    python Test.py --train_save MDFANet_ISIC2017_1e4_bs16_e150_s256 --testsize 256 --test_path /root/autodl-tmp/MDFANet/data/ISIC2017

    ### run with 2018 DSB
    python Train.py --model_name MDFANet --epoch 151 --batchsize 16 --trainsize 256 --train_save MDFANet_DSB_1e4_bs16_e150_s256 --lr 0.0001 --train_path /root/autodl-tmp/MDFANet/data/2018DSB/train --test_path /root/autodl-tmp/MDFANet/data/2018DSB/val/
    sleep 1m
    python Test.py --train_save MDFANet_DSB_1e4_bs16_e150_s256 --testsize 256 --test_path /root/autodl-tmp/MDFANet/data/2018DSB
else
    echo "file not exists"
fi
